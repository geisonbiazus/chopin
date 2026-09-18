/**
 * Starting the agent.
 *
 * Disposable Planner and worker sessions share one runtime. A restarted
 * process reconstructs context from durable Chopin state rather than resuming
 * anything the provider holds — it holds nothing, because the Messages API is
 * stateless and the transcript lives in the session.
 *
 * What each configuration below really declares is a capability boundary: the
 * exact tools a session gets, and the gate that decides, per call, whether it
 * may use one.
 */

import { AnthropicRuntimeClient } from "./anthropic";
import {
	gate,
	PUBLIC_WEB_SEARCH_SERVER,
	PUBLIC_WEB_SEARCH_TOOL,
	publicResearchGate,
	terminalGate,
} from "./permissions";
import { plannerFor } from "./planner";
import { Runtime } from "./runtime";

import type { Config } from "../config";
import type { HostedRepository } from "./repository";
import type {
	AgentSession,
	CurrentToolMetadata,
	CustomAgentConfig,
	SessionConfig,
	Tool,
} from "./types";

export type Agent = {
	session: AgentSession;
	/** Runtime identity used only to delete the disposable session. */
	id: string;
};

/** The tools a planner may call. */
export type Toolbox = { tools: Tool[] };

export type PlannerSession = {
	token: string;
	repository: HostedRepository;
	bootstrap?: string;
	authorize?: () => Promise<boolean>;
};

export type WorkerSession = {
	token: string;
	name: string;
	prompt: string;
	result: Tool;
	maxAiCredits: number;
	authorize?: () => Promise<boolean>;
	onWebSearchDenied?: () => void;
};

const SESSION_CONTROL_TIMEOUT_MS = 10_000;
const MIN_WORKER_AI_CREDITS = 30;
/**
 * How far a public brief may range.
 *
 * Hosted search is the only thing in this process that spends money without a
 * local call to point at, so it gets an explicit ceiling rather than the
 * turn-count bound that covers everything else.
 */
const PUBLIC_WEB_SEARCH_MAX_USES = 12;

function bounded<T>(
	operation: Promise<T>,
	message: string,
	timeoutMs = SESSION_CONTROL_TIMEOUT_MS,
): Promise<T> {
	return new Promise<T>((resolve, reject) => {
		let timer = setTimeout(() => reject(new Error(message)), timeoutMs);
		operation.then(
			value => {
				clearTimeout(timer);
				resolve(value);
			},
			err => {
				clearTimeout(timer);
				reject(err);
			},
		);
	});
}

/**
 * The bound on one session's work.
 *
 * Copilot metered AI credits. This runtime meters model requests, which is the
 * same guarantee — a worker that will not settle is stopped — counted in the
 * unit we actually have. The floor is kept so a caller cannot ask for a
 * session too small to finish anything.
 */
function workerRequestLimit(value: number): number {
	if (!Number.isFinite(value) || value < MIN_WORKER_AI_CREDITS) {
		throw new Error(`Background worker maxAiCredits must be at least ${MIN_WORKER_AI_CREDITS}.`);
	}
	return Math.max(1, Math.floor(value / 2));
}

export function plannerConfiguration(
	config: Pick<Config, "model">,
	toolbox: Toolbox,
	options: PlannerSession,
): SessionConfig {
	let repository = `${options.repository.owner}/${options.repository.name}`;
	let tools = toolbox.tools.map(tool => ({ ...tool, skipPermission: false }));
	return {
		model: config.model,
		streaming: true,
		tools,
		agent: plannerFor(repository),
		systemMessage: [
			`The selected repository is ${repository}. Repository reads must remain inside it.`,
			"More than one person may be in this conversation; their messages are prefixed with the speaker's handle.",
			options.bootstrap ?? "",
		].filter(Boolean).join(" "),
		onPermissionRequest: gate({
			owner: options.repository.owner,
			repository: options.repository.name,
			tools: new Set(tools.map(tool => tool.name)),
			active: options.authorize,
		}),
	};
}

export function workerConfiguration(
	config: Pick<Config, "model">,
	options: WorkerSession,
): SessionConfig {
	let result = { ...options.result, skipPermission: false, isTerminal: true };
	let worker: CustomAgentConfig = {
		name: options.name,
		displayName: "Background worker",
		description: "Executes one registered background job and submits its structured result.",
		prompt: options.prompt,
	};
	return {
		model: config.model,
		streaming: false,
		maxRequests: workerRequestLimit(options.maxAiCredits),
		tools: [result],
		agent: worker,
		onPermissionRequest: terminalGate(result.name, options.authorize),
	};
}

export function publicResearchConfiguration(
	config: Pick<Config, "model">,
	options: WorkerSession,
): SessionConfig {
	let result = { ...options.result, skipPermission: false, isTerminal: true };
	let worker: CustomAgentConfig = {
		name: options.name,
		displayName: "Public research worker",
		description: "Researches public web evidence without private Chopin or repository context.",
		prompt: options.prompt,
	};
	return {
		model: config.model,
		streaming: false,
		maxRequests: workerRequestLimit(options.maxAiCredits),
		tools: [result],
		agent: worker,
		webSearch: { maxUses: PUBLIC_WEB_SEARCH_MAX_USES },
		onPermissionRequest: publicResearchGate(
			result.name,
			options.authorize,
			options.onWebSearchDenied,
		),
	};
}

function connect(apiKey: string) {
	return () => ({
		client: new AnthropicRuntimeClient(apiKey),
		cleanup: () => {},
	});
}

let runtime: Runtime | undefined;
let key: string | undefined;

/**
 * The process-wide runtime, created on first use.
 *
 * The key is read here rather than captured at import so a session opened
 * before configuration has loaded fails with a sentence rather than a
 * confusing authentication error from the API.
 */
function current(config: Pick<Config, "anthropicApiKey">): Runtime {
	if (!config.anthropicApiKey) {
		throw new Error("ANTHROPIC_API_KEY is required to run the hosted agent.");
	}
	if (!runtime || key !== config.anthropicApiKey) {
		runtime = new Runtime(connect(config.anthropicApiKey));
		key = config.anthropicApiKey;
	}
	return runtime;
}

/**
 * Report what the planner can actually call.
 *
 * Costs one line per session, and is never fatal — a diagnostic that can stop
 * a turn is worse than no diagnostic.
 */
function audit(session: AgentSession): void {
	try {
		let names = session.toolMetadata().map(tool => tool.namespacedName || tool.name).sort();
		console.log(`[agent] ${names.length} tools: ${names.join(", ")}`);
	} catch (err) {
		console.warn("[agent] could not read the tool list:", err);
	}
}

export function assertWorkerTools(
	tools: CurrentToolMetadata[] | null | undefined,
	expected: string,
	publicWeb = false,
): void {
	let values = tools ?? [];
	let result = values.filter(tool =>
		tool.name === expected
		&& !tool.mcpServerName
		&& !tool.mcpToolName
		&& (!tool.namespacedName || tool.namespacedName === `custom:${expected}`)
	);
	let web = values.filter(tool =>
		tool.mcpServerName === PUBLIC_WEB_SEARCH_SERVER
		&& tool.mcpToolName === PUBLIC_WEB_SEARCH_TOOL
	);
	let matches = values.length === (publicWeb ? 2 : 1)
		&& result.length === 1
		&& web.length === (publicWeb ? 1 : 0);
	if (!matches) {
		let wanted = [
			`custom:${expected}`,
			...(publicWeb ? [`mcp:${PUBLIC_WEB_SEARCH_TOOL}`] : []),
		];
		let received = values.map(tool =>
			tool.namespacedName
				?? (tool.mcpServerName || tool.mcpToolName
					? `mcp:${tool.mcpServerName ?? "unknown"}-${tool.mcpToolName ?? tool.name}`
					: `local:${tool.name}`)
		).sort();
		throw new Error(
			`Background worker capability audit failed: expected ${wanted.join(", ")}, received ${
				received.length > 0 ? received.join(", ") : "none"
			}.`,
		);
	}
}

/** Create a disposable session authenticated and scoped to one owner and repository. */
export async function openPlanner(
	config: Pick<Config, "agent" | "model" | "anthropicApiKey">,
	toolbox: Toolbox,
	options: PlannerSession,
): Promise<Agent> {
	if (!config.agent) throw new Error("The hosted agent is disabled.");
	let live = current(config);
	let session = await live.open(plannerConfiguration(config, toolbox, options));
	try {
		audit(session);
		return { session, id: session.sessionId };
	} catch (err) {
		await live.discard(session).catch(() => {});
		throw err;
	}
}

/** Create a disposable isolated session for one registered background attempt. */
export async function openWorker(
	config: Pick<Config, "agent" | "model" | "anthropicApiKey">,
	options: WorkerSession,
): Promise<Agent> {
	if (!config.agent) throw new Error("The hosted agent is disabled.");
	let live = current(config);
	let session = await live.open(workerConfiguration(config, options));
	try {
		assertWorkerTools(session.toolMetadata(), options.result.name);
		return { session, id: session.sessionId };
	} catch (err) {
		await live.discard(session).catch(() => {});
		throw err;
	}
}

/** Create a public-web worker with no private document or repository capabilities. */
export async function openPublicResearchWorker(
	config: Pick<Config, "agent" | "model" | "anthropicApiKey">,
	options: WorkerSession,
): Promise<Agent> {
	if (!config.agent) throw new Error("The hosted agent is disabled.");
	let live = current(config);
	let session = await live.open(publicResearchConfiguration(config, options));
	try {
		assertWorkerTools(session.toolMetadata(), options.result.name, true);
		return { session, id: session.sessionId };
	} catch (err) {
		await live.discard(session).catch(() => {});
		throw err;
	}
}

export async function discard(agent: Agent): Promise<void> {
	// A session the runtime never owned — or one outliving the runtime that
	// opened it — still has to be told to let go.
	let owned = false;
	if (runtime) {
		try {
			owned = await runtime.discard(agent.session);
		} catch {
			return;
		}
	}
	if (!owned) await agent.session.disconnect().catch(() => {});
}

/** Bound an abort so runtime shutdown can still force a wedged session down. */
export async function abort(agent: Agent): Promise<void> {
	await bounded(
		Promise.resolve().then(() => agent.session.abort()),
		`Agent session ${agent.id} abort timed out.`,
	).catch(() => {});
}

/** Stop waiting for an opening session, and dispose it if it arrives later. */
export async function settle(opening: Promise<Agent>): Promise<Agent | undefined> {
	let expired = false;
	let watched = opening.then(agent => {
		if (expired) void discard(agent);
		return agent;
	}, () => undefined);
	let opened = await bounded(watched, "Agent session opening timed out.").catch(() => undefined);
	expired = true;
	return opened;
}

/** Close every remaining session and let go of the runtime. */
export async function shutdown(): Promise<void> {
	let live = runtime;
	runtime = undefined;
	key = undefined;
	if (live) await live.shutdown();
}

/** Only for tests: drop the memoised runtime between cases. */
export function reset(): void {
	runtime = undefined;
	key = undefined;
}
