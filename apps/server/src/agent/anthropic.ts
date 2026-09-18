/**
 * The runtime.
 *
 * One session is one conversation with Anthropic's Messages API, driven here
 * rather than by a CLI: the loop asks the model, gates every tool call it
 * proposes, runs the ones that survive, and feeds the results back until the
 * model stops asking. What it emits is the event stream the rest of the server
 * already reads, so chat, jobs and the planner never learn which provider
 * answered.
 *
 * The API is stateless, so the transcript lives here for the life of the
 * session and dies with it. A restarted process reconstructs context from
 * durable Chopin state, exactly as before.
 */

import Anthropic from "@anthropic-ai/sdk";

import { PUBLIC_WEB_SEARCH_SERVER, PUBLIC_WEB_SEARCH_TOOL } from "./permissions";

import type {
	AgentSession,
	CitableSource,
	CurrentToolMetadata,
	SessionConfig,
	SessionEvent,
	SessionListener,
	Tool,
} from "./types";

/**
 * Room to answer in.
 *
 * Streaming turns are the conversational ones and get the larger ceiling
 * because a plan revision is long and there is no request timeout to fear;
 * a buffered worker turn is a bounded structured result and does not need it.
 */
const STREAMING_MAX_TOKENS = 64_000;
const BUFFERED_MAX_TOKENS = 16_000;
const DEFAULT_MAX_REQUESTS = 64;

/** Hosted search is the one tool Anthropic runs for us rather than we for it. */
const WEB_SEARCH_TOOL_TYPE = "web_search_20260209";

function message(err: unknown): string {
	return err instanceof Error ? err.message : String(err);
}

/** A tool result is text; a failure is text that says so. */
function resultText(value: unknown): string {
	if (typeof value === "string") return value;
	if (value === undefined) return "null";
	try {
		return JSON.stringify(value) ?? "null";
	} catch {
		return String(value);
	}
}

function declaration(tool: Tool, eager: boolean): Anthropic.ToolUnion {
	let schema = tool.parameters ?? { type: "object", properties: {}, additionalProperties: false };
	return {
		name: tool.name,
		...(tool.description ? { description: tool.description } : {}),
		input_schema: schema as Anthropic.Tool.InputSchema,
		// Large arguments — a plan revision is the common case — arrive as they
		// are written rather than in one burst once the server has buffered
		// them. The handlers validate their own arguments, which is what makes
		// this safe: a truncated input fails validation like any other bad one.
		...(eager ? { eager_input_streaming: true } : {}),
	} as Anthropic.ToolUnion;
}

/**
 * What the session ended up holding.
 *
 * Hosted web search is reported with the server and tool names the audits and
 * research metrics already look for, because from their side it is still "the
 * one read-only web search tool and nothing else".
 */
function metadata(config: SessionConfig): CurrentToolMetadata[] {
	let tools = config.tools.map(tool => ({
		name: tool.name,
		namespacedName: `custom:${tool.name}`,
	}));
	if (!config.webSearch) return tools;
	return [...tools, {
		name: PUBLIC_WEB_SEARCH_TOOL,
		mcpServerName: PUBLIC_WEB_SEARCH_SERVER,
		mcpToolName: PUBLIC_WEB_SEARCH_TOOL,
	}];
}

/** Everything the model said it read, flattened to what provenance needs. */
function citations(content: Anthropic.ContentBlock[]): CitableSource[] {
	let sources: CitableSource[] = [];
	let seen = new Set<string>();
	for (let block of content) {
		if (block.type !== "text" || !block.citations) continue;
		for (let citation of block.citations) {
			let url = "url" in citation && typeof citation.url === "string" ? citation.url : undefined;
			let title = "title" in citation && typeof citation.title === "string"
				? citation.title
				: undefined;
			let key = url ?? title;
			if (!key || seen.has(key)) continue;
			seen.add(key);
			sources.push({ ...(url ? { url } : {}), ...(title ? { title } : {}) });
		}
	}
	return sources;
}

function searchResults(block: Anthropic.ContentBlock): CitableSource[] {
	if (block.type !== "web_search_tool_result" || !Array.isArray(block.content)) return [];
	return block.content.flatMap(result =>
		result.type === "web_search_result"
			? [{ url: result.url, ...(result.title ? { title: result.title } : {}) }]
			: []
	);
}

export class AnthropicSession implements AgentSession {
	readonly sessionId = crypto.randomUUID();

	#client: Anthropic;
	#config: SessionConfig;
	#listeners = new Set<SessionListener>();
	#transcript: Anthropic.MessageParam[] = [];
	#controller?: AbortController;
	#requests = 0;
	/**
	 * Hosted searches counted across the session rather than per message.
	 *
	 * Research dedupes by this index, and a per-message block index would
	 * repeat on the second request and silently under-count.
	 */
	#hostedCalls = 0;
	#closed = false;
	#running?: Promise<void>;

	constructor(client: Anthropic, config: SessionConfig) {
		this.#client = client;
		this.#config = config;
	}

	on(listener: SessionListener): () => void {
		this.#listeners.add(listener);
		return () => this.#listeners.delete(listener);
	}

	toolMetadata(): CurrentToolMetadata[] {
		return metadata(this.#config);
	}

	/**
	 * Accept a message and start the turn.
	 *
	 * Resolves when the message is accepted, not when the turn is over: the
	 * work happens afterwards over events, and every caller waits for
	 * `session.idle` or `session.error` rather than for this.
	 */
	async send({ prompt }: { prompt: string }): Promise<void> {
		if (this.#closed) throw new Error("The agent session is closed.");
		if (this.#running) throw new Error("The agent session is already answering.");
		this.#transcript.push({ role: "user", content: prompt });
		this.#running = this.#turn().finally(() => {
			this.#running = undefined;
		});
		// A rejection is reported as `session.error` inside the turn; nothing
		// awaits this promise, so it must never surface as unhandled.
		void this.#running.catch(() => {});
	}

	async abort(): Promise<void> {
		this.#controller?.abort(new Error("The agent turn was aborted."));
	}

	async disconnect(): Promise<void> {
		this.#closed = true;
		this.#controller?.abort(new Error("The agent session was closed."));
		this.#listeners.clear();
	}

	#emit(event: SessionEvent): void {
		// Snapshot: a listener that releases itself while reacting must not
		// change what the rest of this dispatch visits.
		let listeners = [...this.#listeners];
		for (let listener of listeners) {
			try {
				listener(event);
			} catch (err) {
				console.error("[agent] a session listener threw:", err);
			}
		}
	}

	async #turn(): Promise<void> {
		let controller = new AbortController();
		this.#controller = controller;
		try {
			await this.#loop(controller.signal);
			this.#emit({ type: "session.idle", data: {} });
		} catch (err) {
			this.#emit({ type: "session.error", data: { message: message(err) } });
		} finally {
			this.#controller = undefined;
		}
	}

	async #loop(signal: AbortSignal): Promise<void> {
		let limit = this.#config.maxRequests ?? DEFAULT_MAX_REQUESTS;
		while (true) {
			if (signal.aborted) throw signal.reason ?? new Error("The agent turn was aborted.");
			if (this.#requests >= limit) {
				throw new Error(`The agent session reached its bound of ${limit} model requests.`);
			}
			this.#requests++;

			let reply = await this.#ask(signal);
			this.#transcript.push({ role: "assistant", content: reply.content });
			this.#report(reply);

			if (reply.stop_reason === "refusal") {
				throw new Error("The model declined to answer this request.");
			}
			// A truncated reply may carry a half-written tool call. Running it
			// would act on arguments the model never finished choosing.
			if (reply.stop_reason === "max_tokens") {
				throw new Error("The model's reply exceeded its output bound.");
			}
			let calls = reply.content.filter((block): block is Anthropic.ToolUseBlock =>
				block.type === "tool_use"
			);
			if (calls.length === 0) return;

			let { results, terminated } = await this.#invoke(calls);
			this.#transcript.push({ role: "user", content: results });
			if (terminated) return;
		}
	}

	async #ask(signal: AbortSignal): Promise<Anthropic.Message> {
		let config = this.#config;
		let system = [config.agent.prompt, config.systemMessage].filter(Boolean).join("\n\n");
		let tools: Anthropic.ToolUnion[] = config.tools.map(tool =>
			declaration(tool, config.streaming)
		);
		if (config.webSearch) {
			tools.push({
				type: WEB_SEARCH_TOOL_TYPE,
				name: PUBLIC_WEB_SEARCH_TOOL,
				max_uses: config.webSearch.maxUses,
			} as Anthropic.ToolUnion);
		}

		let request: Anthropic.MessageCreateParamsStreaming = {
			model: config.model,
			max_tokens: config.streaming ? STREAMING_MAX_TOKENS : BUFFERED_MAX_TOKENS,
			system,
			messages: this.#transcript,
			thinking: { type: "adaptive" },
			...(tools.length > 0 ? { tools } : {}),
			stream: true,
		};

		// Streamed either way: the ceilings here are high enough that a
		// buffered request would be racing an HTTP timeout. Only the deltas
		// are conditional, because only a person watching wants them.
		let stream = this.#client.messages.stream(request, { signal });
		if (config.streaming) {
			let messageId = "";
			stream.on("streamEvent", event => {
				if (event.type === "message_start") messageId = event.message.id;
				else if (
					event.type === "content_block_delta" && event.delta.type === "text_delta"
					&& event.delta.text
				) {
					this.#emit({
						type: "assistant.message_delta",
						data: { messageId, deltaContent: event.delta.text },
					});
				}
			});
		}
		return await stream.finalMessage();
	}

	/** Turn a finished reply into what the room and the metrics read. */
	#report(reply: Anthropic.Message): void {
		for (let block of reply.content) {
			if (block.type === "server_tool_use" && block.name === PUBLIC_WEB_SEARCH_TOOL) {
				this.#emit({
					type: "assistant.server_tool_progress",
					data: { kind: "web_search", outputIndex: this.#hostedCalls++, status: "in_progress" },
				});
			} else if (block.type === "web_search_tool_result") {
				// The result block's own content says whether the search ran:
				// a list of results on success, a single error object on failure.
				let failed = !Array.isArray(block.content);
				this.#emit({
					type: "assistant.server_tool_progress",
					data: {
						kind: "web_search",
						outputIndex: Math.max(0, this.#hostedCalls - 1),
						status: failed ? "in_progress" : "completed",
					},
				});
			}
		}

		let text = reply.content
			.filter((block): block is Anthropic.TextBlock => block.type === "text")
			.map(block => block.text)
			.join("");
		let sources = [
			...citations(reply.content),
			...reply.content.flatMap(searchResults),
		];
		if (!text.trim() && sources.length === 0) return;
		this.#emit({
			type: "assistant.message",
			data: {
				messageId: reply.id,
				content: text,
				...(sources.length > 0 ? { citations: { sources } } : {}),
			},
		});
	}

	/**
	 * Run the calls that are allowed to run.
	 *
	 * Every call is gated, and a refusal is reported rather than silently
	 * dropped: a boundary nobody can see being hit is indistinguishable from a
	 * tool the agent never had. A refused call still answers, because the API
	 * requires a result for every call in the message.
	 */
	async #invoke(
		calls: Anthropic.ToolUseBlock[],
	): Promise<{ results: Anthropic.ToolResultBlockParam[]; terminated: boolean }> {
		let terminated = false;
		let results = await Promise.all(calls.map(async call => {
			let tool = this.#config.tools.find(candidate => candidate.name === call.name);
			if (!tool) {
				return this.#refuse(call, `${call.name} is not available to this agent.`);
			}

			let decision = await this.#config.onPermissionRequest({
				kind: "custom-tool",
				toolName: call.name,
				args: call.input,
				toolCallId: call.id,
			}).catch(err => ({ kind: "reject", feedback: message(err) } as const));

			if (decision.kind !== "approve-once") {
				return this.#refuse(call, decision.feedback);
			}

			this.#emit({
				type: "tool.execution_start",
				data: { toolCallId: call.id, toolName: call.name, arguments: call.input },
			});

			try {
				let produced = await tool.handler?.(call.input, {
					toolCallId: call.id,
					sessionId: this.sessionId,
					toolName: call.name,
					arguments: call.input,
				});
				let content = resultText(produced);
				this.#emit({
					type: "tool.execution_complete",
					data: { toolCallId: call.id, success: true, result: { content } },
				});
				if (tool.isTerminal) terminated = true;
				return {
					type: "tool_result" as const,
					tool_use_id: call.id,
					content,
				};
			} catch (err) {
				// A throwing handler leaves the loop running so the model can
				// read the error and try again; only a clean terminal call ends
				// the turn.
				this.#emit({
					type: "tool.execution_complete",
					data: { toolCallId: call.id, success: false, error: message(err) },
				});
				return {
					type: "tool_result" as const,
					tool_use_id: call.id,
					content: `Error: ${message(err)}`,
					is_error: true,
				};
			}
		}));
		return { results, terminated };
	}

	#refuse(call: Anthropic.ToolUseBlock, feedback: string): Anthropic.ToolResultBlockParam {
		this.#emit({
			type: "permission.completed",
			data: { toolCallId: call.id, result: { kind: "denied", feedback } },
		});
		return {
			type: "tool_result",
			tool_use_id: call.id,
			content: feedback,
			is_error: true,
		};
	}
}

/**
 * The session factory the Runtime owns.
 *
 * There is no child process to start or stop any more, so most of this is
 * bookkeeping — but the Runtime's guarantees (bounded shutdown, no session
 * outliving its generation) are worth keeping, and they are expressed in terms
 * of a client that can be asked to let go.
 */
export class AnthropicRuntimeClient {
	#client: Anthropic;
	#sessions = new Map<string, AnthropicSession>();

	constructor(apiKey: string) {
		this.#client = new Anthropic({ apiKey });
	}

	async start(): Promise<void> {}

	async createSession(config: SessionConfig): Promise<AgentSession> {
		let session = new AnthropicSession(this.#client, config);
		this.#sessions.set(session.sessionId, session);
		return session;
	}

	async deleteSession(sessionId: string): Promise<void> {
		this.#sessions.delete(sessionId);
	}

	async stop(): Promise<Error[]> {
		let errors: Error[] = [];
		let sessions = [...this.#sessions.values()];
		for (let session of sessions) {
			try {
				await session.disconnect();
			} catch (err) {
				errors.push(err instanceof Error ? err : new Error(String(err)));
			}
		}
		this.#sessions.clear();
		return errors;
	}

	async forceStop(): Promise<void> {
		this.#sessions.clear();
	}
}
