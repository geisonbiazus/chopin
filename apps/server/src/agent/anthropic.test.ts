import { describe, expect, it } from "bun:test";

import { AnthropicSession } from "./anthropic";

import type Anthropic from "@anthropic-ai/sdk";
import type { SessionConfig, SessionEvent, Tool } from "./types";

type Reply = {
	content: Anthropic.ContentBlock[];
	stop_reason?: Anthropic.Message["stop_reason"];
	id?: string;
};

type Recorded = { system: string; tools: unknown[]; messages: Anthropic.MessageParam[] };

/**
 * A model that says exactly what a case needs, in order.
 *
 * Deltas are derived from the text blocks of each reply rather than scripted
 * separately, so a case cannot accidentally stream something the final message
 * does not contain.
 */
function model(replies: Reply[]) {
	let seen: Recorded[] = [];
	let client = {
		messages: {
			stream(request: Record<string, unknown>) {
				seen.push({
					system: String(request.system ?? ""),
					tools: (request.tools as unknown[]) ?? [],
					messages: structuredClone(request.messages as Anthropic.MessageParam[]),
				});
				let reply = replies.shift();
				if (!reply) throw new Error("the model was asked more times than the case scripted");
				let listeners: ((event: unknown) => void)[] = [];
				let message = {
					id: reply.id ?? `msg_${seen.length}`,
					content: reply.content,
					stop_reason: reply.stop_reason ?? "end_turn",
				} as Anthropic.Message;
				return {
					on(_name: string, listener: (event: unknown) => void) {
						listeners.push(listener);
					},
					async finalMessage() {
						for (let listener of listeners) {
							listener({ type: "message_start", message: { id: message.id } });
							for (let block of reply.content) {
								if (block.type !== "text") continue;
								listener({
									type: "content_block_delta",
									delta: { type: "text_delta", text: block.text },
								});
							}
						}
						return message;
					},
				};
			},
		},
	};
	return { client: client as unknown as Anthropic, seen };
}

function text(value: string): Anthropic.ContentBlock {
	return { type: "text", text: value, citations: null } as Anthropic.ContentBlock;
}

function call(id: string, name: string, input: unknown = {}): Anthropic.ContentBlock {
	return { type: "tool_use", id, name, input } as Anthropic.ContentBlock;
}

function settings(overrides: Partial<SessionConfig> = {}): SessionConfig {
	return {
		model: "claude-opus-5",
		streaming: true,
		agent: { name: "chopin-plan", prompt: "You are the planner." },
		tools: [],
		onPermissionRequest: async () => ({ kind: "approve-once" }),
		...overrides,
	};
}

/** Run one turn and hand back everything it said, in order. */
async function turn(
	client: Anthropic,
	config: SessionConfig,
	prompt = "hello",
): Promise<SessionEvent[]> {
	let session = new AnthropicSession(client, config);
	let events: SessionEvent[] = [];
	let done = Promise.withResolvers<void>();
	session.on(event => {
		events.push(event);
		if (event.type === "session.idle" || event.type === "session.error") done.resolve();
	});
	await session.send({ prompt });
	await done.promise;
	return events;
}

describe("anthropic session", () => {
	it("streams a plain reply and ends the turn once", async () => {
		let { client, seen } = model([{ content: [text("Here is the plan.")] }]);
		let events = await turn(client, settings());

		expect(events.map(event => event.type)).toEqual([
			"assistant.message_delta",
			"assistant.message",
			"session.idle",
		]);
		let delta = events[0] as Extract<SessionEvent, { type: "assistant.message_delta" }>;
		let message = events[1] as Extract<SessionEvent, { type: "assistant.message" }>;
		expect(delta.data.deltaContent).toBe("Here is the plan.");
		// The deltas and the message must agree on the id, or the room files
		// the finished message as a second entry beside the streaming one.
		expect(delta.data.messageId).toBe(message.data.messageId);
		expect(message.data.content).toBe("Here is the plan.");
		expect(seen[0]?.system).toContain("You are the planner.");
	});

	it("appends the system message to the agent's own prompt", async () => {
		let { client, seen } = model([{ content: [text("ok")] }]);
		await turn(client, settings({ systemMessage: "The selected repository is octo-org/score." }));
		expect(seen[0]?.system).toContain("You are the planner.");
		expect(seen[0]?.system).toContain("octo-org/score");
	});

	it("runs an approved tool and feeds its result back", async () => {
		let received: unknown;
		let tool: Tool = {
			name: "read_plan",
			parameters: { type: "object", properties: {}, additionalProperties: false },
			handler: args => {
				received = args;
				return "revision 4";
			},
		};
		let { client, seen } = model([
			{ content: [call("call_1", "read_plan", { scope: "all" })], stop_reason: "tool_use" },
			{ content: [text("The plan is at revision 4.")] },
		]);
		let events = await turn(client, settings({ tools: [tool] }));

		expect(received).toEqual({ scope: "all" });
		expect(events.map(event => event.type)).toEqual([
			"tool.execution_start",
			"tool.execution_complete",
			"assistant.message_delta",
			"assistant.message",
			"session.idle",
		]);
		let complete = events[1] as Extract<SessionEvent, { type: "tool.execution_complete" }>;
		expect(complete.data).toMatchObject({
			toolCallId: "call_1",
			success: true,
			result: { content: "revision 4" },
		});
		// The second request carries the first reply and its result.
		expect(seen[1]?.messages).toHaveLength(3);
		expect(seen[1]?.messages[2]).toMatchObject({
			role: "user",
			content: [{ type: "tool_result", tool_use_id: "call_1", content: "revision 4" }],
		});
	});

	it("reports a refused tool and never runs it", async () => {
		let ran = 0;
		let tool: Tool = { name: "edit_plan", handler: () => ++ran };
		let { client, seen } = model([
			{ content: [call("call_1", "edit_plan")], stop_reason: "tool_use" },
			{ content: [text("I cannot edit the plan.")] },
		]);
		let events = await turn(
			client,
			settings({
				tools: [tool],
				onPermissionRequest: async () => ({ kind: "reject", feedback: "not available" }),
			}),
		);

		expect(ran).toBe(0);
		expect(events.map(event => event.type)).toEqual([
			"permission.completed",
			"assistant.message_delta",
			"assistant.message",
			"session.idle",
		]);
		expect(events[0]).toMatchObject({
			data: { toolCallId: "call_1", result: { kind: "denied", feedback: "not available" } },
		});
		// The refusal is still answered: the API requires a result per call,
		// and the model is told why so it can work with the boundary.
		expect(seen[1]?.messages[2]).toMatchObject({
			content: [{ type: "tool_result", tool_use_id: "call_1", is_error: true }],
		});
	});

	it("refuses a tool the session was never given", async () => {
		let { client } = model([
			{ content: [call("call_1", "bash")], stop_reason: "tool_use" },
			{ content: [text("understood")] },
		]);
		let events = await turn(client, settings());
		expect(events[0]).toMatchObject({
			type: "permission.completed",
			data: { result: { kind: "denied", feedback: "bash is not available to this agent." } },
		});
	});

	it("keeps the turn running when a handler throws", async () => {
		let tool: Tool = {
			name: "read_plan",
			handler: () => {
				throw new Error("the plan moved on");
			},
		};
		let { client, seen } = model([
			{ content: [call("call_1", "read_plan")], stop_reason: "tool_use" },
			{ content: [text("I will read it again.")] },
		]);
		let events = await turn(client, settings({ tools: [tool] }));

		let complete = events[1] as Extract<SessionEvent, { type: "tool.execution_complete" }>;
		expect(complete.data).toMatchObject({ success: false, error: "the plan moved on" });
		expect(events.at(-1)?.type).toBe("session.idle");
		expect(seen[1]?.messages[2]).toMatchObject({
			content: [{ type: "tool_result", is_error: true, content: "Error: the plan moved on" }],
		});
	});

	it("ends the turn on a terminal tool without asking again", async () => {
		let tool: Tool = { name: "submit_job_result", isTerminal: true, handler: () => "recorded" };
		let { client, seen } = model([
			{ content: [call("call_1", "submit_job_result")], stop_reason: "tool_use" },
		]);
		let events = await turn(client, settings({ streaming: false, tools: [tool] }));

		expect(seen).toHaveLength(1);
		expect(events.map(event => event.type)).toEqual([
			"tool.execution_start",
			"tool.execution_complete",
			"session.idle",
		]);
	});

	it("streams deltas only when the session asked for them", async () => {
		let { client } = model([{ content: [text("quiet")] }]);
		let events = await turn(client, settings({ streaming: false }));
		expect(events.map(event => event.type)).toEqual(["assistant.message", "session.idle"]);
	});

	it("declares client tools as eager only while streaming", async () => {
		let tool: Tool = { name: "read_plan", handler: () => "ok" };
		let streamed = model([{ content: [text("ok")] }]);
		await turn(streamed.client, settings({ tools: [tool] }));
		expect(streamed.seen[0]?.tools[0]).toMatchObject({ eager_input_streaming: true });

		let buffered = model([{ content: [text("ok")] }]);
		await turn(buffered.client, settings({ streaming: false, tools: [tool] }));
		expect(buffered.seen[0]?.tools[0]).not.toMatchObject({ eager_input_streaming: true });
	});

	it("stops rather than running a tool call the model did not finish writing", async () => {
		let ran = 0;
		let tool: Tool = { name: "edit_plan", handler: () => ++ran };
		let { client } = model([
			{ content: [call("call_1", "edit_plan", { truncated: true })], stop_reason: "max_tokens" },
		]);
		let events = await turn(client, settings({ tools: [tool] }));

		expect(ran).toBe(0);
		expect(events.at(-1)).toMatchObject({
			type: "session.error",
			data: { message: "The model's reply exceeded its output bound." },
		});
	});

	it("stops a session that will not settle", async () => {
		let tool: Tool = { name: "read_plan", handler: () => "again" };
		let { client } = model([
			{ content: [call("call_1", "read_plan")], stop_reason: "tool_use" },
			{ content: [call("call_2", "read_plan")], stop_reason: "tool_use" },
			{ content: [call("call_3", "read_plan")], stop_reason: "tool_use" },
		]);
		let events = await turn(client, settings({ tools: [tool], maxRequests: 2 }));

		expect(events.at(-1)).toMatchObject({
			type: "session.error",
			data: { message: "The agent session reached its bound of 2 model requests." },
		});
	});

	it("reports a provider failure as one ended turn", async () => {
		let client = {
			messages: {
				stream() {
					throw new Error("401 authentication_error");
				},
			},
		} as unknown as Anthropic;
		let events = await turn(client, settings());
		expect(events).toHaveLength(1);
		expect(events[0]).toMatchObject({
			type: "session.error",
			data: { message: "401 authentication_error" },
		});
	});

	it("declares hosted search only for a session given it, and counts its use", async () => {
		let search = { type: "server_tool_use", id: "srv_1", name: "web_search" };
		let results = {
			type: "web_search_tool_result",
			tool_use_id: "srv_1",
			content: [{ type: "web_search_result", url: "https://example.com/a", title: "A" }],
		};
		let cited = {
			type: "text",
			text: "Evidence says so.",
			citations: [{ type: "web_search_result_location", url: "https://example.com/a", title: "A" }],
		};
		let { client, seen } = model([{
			content: [search, results, cited] as unknown as Anthropic.ContentBlock[],
		}]);
		let events = await turn(
			client,
			settings({ streaming: false, webSearch: { maxUses: 12 } }),
		);

		expect(seen[0]?.tools).toEqual([
			{ type: "web_search_20260209", name: "web_search", max_uses: 12 },
		]);
		expect(events.map(event => event.type)).toEqual([
			"assistant.server_tool_progress",
			"assistant.server_tool_progress",
			"assistant.message",
			"session.idle",
		]);
		expect(events[0]).toMatchObject({ data: { outputIndex: 0, status: "in_progress" } });
		expect(events[1]).toMatchObject({ data: { outputIndex: 0, status: "completed" } });
		let message = events[2] as Extract<SessionEvent, { type: "assistant.message" }>;
		expect(message.data.citations?.sources).toContainEqual({
			url: "https://example.com/a",
			title: "A",
		});
	});

	it("does not count a failed hosted search as completed", async () => {
		let search = { type: "server_tool_use", id: "srv_1", name: "web_search" };
		let failure = {
			type: "web_search_tool_result",
			tool_use_id: "srv_1",
			content: { type: "web_search_tool_result_error", error_code: "max_uses_exceeded" },
		};
		let { client } = model([{
			content: [
				search,
				failure,
				text("I could not search."),
			] as unknown as Anthropic.ContentBlock[],
		}]);
		let events = await turn(client, settings({ streaming: false, webSearch: { maxUses: 1 } }));
		let progress = events.filter(event => event.type === "assistant.server_tool_progress");
		expect(progress).toHaveLength(2);
		expect(
			progress.every(event =>
				event.type === "assistant.server_tool_progress" && event.data.status === "in_progress"
			),
		).toBe(true);
	});

	it("reports the tools a worker session actually holds", () => {
		let session = new AnthropicSession(
			model([]).client,
			settings({ tools: [{ name: "submit_job_result" }], webSearch: { maxUses: 4 } }),
		);
		expect(session.toolMetadata()).toEqual([
			{ name: "submit_job_result", namespacedName: "custom:submit_job_result" },
			{
				name: "web_search",
				mcpServerName: "github-mcp-server",
				mcpToolName: "web_search",
			},
		]);
	});

	it("refuses to answer twice at once and refuses a closed session", async () => {
		let { client } = model([{ content: [text("one")] }]);
		let session = new AnthropicSession(client, settings());
		let done = Promise.withResolvers<void>();
		session.on(event => {
			if (event.type === "session.idle") done.resolve();
		});
		await session.send({ prompt: "first" });
		expect(session.send({ prompt: "second" })).rejects.toThrow("already answering");
		await done.promise;

		await session.disconnect();
		expect(session.send({ prompt: "third" })).rejects.toThrow("closed");
	});
});
