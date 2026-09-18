/**
 * The agent vocabulary.
 *
 * These types were once the Copilot SDK's. They are ours now, because the
 * runtime is: the shapes that survive here are the ones the rest of the server
 * already speaks — tools, permission decisions, and the event stream a turn
 * produces — and keeping their names means chat, jobs and the planner did not
 * have to be rewritten to change which model answers.
 *
 * Deliberately narrower than what Anthropic's API can express. A field exists
 * here because something reads it.
 */

/** What a handler is told about the call it is answering. */
export type ToolInvocation = {
	toolCallId: string;
	sessionId?: string;
	toolName?: string;
	/** The raw arguments, as handed to the handler. */
	arguments?: unknown;
};

export type ToolHandler<TArgs = unknown> = (
	args: TArgs,
	invocation: ToolInvocation,
) => Promise<unknown> | unknown;

/** One capability offered to the model, with the handler that answers it. */
export type Tool<TArgs = unknown> = {
	name: string;
	description?: string;
	/** JSON Schema. Absent means the tool takes no arguments. */
	parameters?: Record<string, unknown>;
	handler?: ToolHandler<TArgs>;
	/** Skips the permission gate. Chopin never sets this: everything is gated. */
	skipPermission?: boolean;
	/** A successful call ends the turn instead of feeding the result back. */
	isTerminal?: boolean;
};

/**
 * A named persona with its own prompt.
 *
 * Copilot discovered these from the runtime; here one is simply the system
 * prompt a session is opened with, which is what it always meant.
 */
export type CustomAgentConfig = {
	name: string;
	displayName?: string;
	description?: string;
	prompt: string;
};

export type PermissionRequest =
	| { kind: "custom-tool"; toolName: string; args?: unknown; toolCallId?: string }
	| {
		kind: "mcp";
		serverName: string;
		toolName: string;
		readOnly: boolean;
		args?: unknown;
		toolCallId?: string;
	}
	| { kind: "url"; url: string; toolCallId?: string }
	| { kind: "shell"; command: string; toolCallId?: string }
	| { kind: "file"; path: string; toolCallId?: string };

export type PermissionRequestResult =
	| { kind: "approve-once" }
	| { kind: "reject"; feedback: string };

export type PermissionHandler = (
	request: PermissionRequest,
) => Promise<PermissionRequestResult>;

/** What a session reports about the tools it actually ended up with. */
export type CurrentToolMetadata = {
	name: string;
	namespacedName?: string;
	mcpServerName?: string;
	mcpToolName?: string;
};

/** A source the model says it read, as carried on a message or tool result. */
export type CitableSource = {
	url?: string;
	title?: string;
};

export type ToolResultPayload = {
	content?: string;
	citableSources?: CitableSource[];
};

/**
 * Everything a turn can say.
 *
 * `session.idle` ends a turn and `session.error` ends it badly; exactly one of
 * the two is emitted per turn, and every consumer treats either as the end.
 */
export type SessionEvent =
	| { type: "session.idle"; data: Record<string, never> }
	| { type: "session.error"; data: { message: string } }
	| { type: "assistant.message_delta"; data: { messageId: string; deltaContent: string } }
	| {
		type: "assistant.message";
		data: {
			messageId: string;
			content: string;
			citations?: { sources: CitableSource[] };
		};
	}
	| {
		type: "assistant.server_tool_progress";
		data: { kind: "web_search"; outputIndex: number; status: "in_progress" | "completed" };
	}
	| {
		type: "tool.execution_start";
		data: {
			toolCallId: string;
			toolName: string;
			arguments?: unknown;
			mcpServerName?: string;
			mcpToolName?: string;
		};
	}
	| {
		type: "tool.execution_complete";
		data: {
			toolCallId: string;
			success: boolean;
			result?: ToolResultPayload;
			error?: unknown;
			mcpServerName?: string;
			mcpToolName?: string;
		};
	}
	| {
		type: "permission.completed";
		data: {
			toolCallId?: string;
			result: { kind: "approved" | "denied"; feedback?: string };
		};
	};

export type SessionListener = (event: SessionEvent) => void;

/**
 * Hosted web search, run by Anthropic rather than by us.
 *
 * Public research is the only thing that gets it, and `maxUses` is the bound
 * that stops a runaway brief from becoming a runaway bill.
 */
export type WebSearchConfig = {
	maxUses: number;
};

export type SessionConfig = {
	model: string;
	/** Emits `assistant.message_delta` while the model writes. */
	streaming: boolean;
	agent: CustomAgentConfig;
	/** Appended to the agent's own prompt. */
	systemMessage?: string;
	tools: Tool[];
	webSearch?: WebSearchConfig;
	onPermissionRequest: PermissionHandler;
	/**
	 * How many model requests one session may make.
	 *
	 * Copilot counted AI credits; we count round trips, which is the same
	 * guard against a worker that will not stop, expressed in the unit this
	 * runtime actually has.
	 */
	maxRequests?: number;
};

/** The subset of a live session the rest of the server drives. */
export type AgentSession = {
	sessionId: string;
	on: (listener: SessionListener) => () => void;
	send: (message: { prompt: string }) => Promise<void>;
	abort: () => Promise<void>;
	disconnect: () => Promise<void>;
	/** The tools the session ended up with, for the capability audits. */
	toolMetadata: () => CurrentToolMetadata[];
};
