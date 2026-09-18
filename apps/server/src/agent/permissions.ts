/**
 * Capability boundary for the shared, repository-scoped agent runtime.
 *
 * Every tool call is gated, and the gate is an allow-list: a request that is
 * not a named custom tool this session was given is refused, whatever it is.
 * Nothing here can widen what a session holds — the session's tool list does
 * that — so a gate that recognises fewer shapes is a stricter one.
 *
 * Repository scope used to be argued about here, because the pull request
 * tools lived on a shared MCP server that would happily read somebody else's
 * code if asked. They are ours now and close over the channel's repository, so
 * there is no argument left to inspect.
 */

import type { PermissionHandler, PermissionRequest, PermissionRequestResult } from "./types";

/**
 * How hosted web search identifies itself.
 *
 * Kept as the label the research metrics and worker audits already match on:
 * search is Anthropic's to run now, not a server we connect to, but from the
 * outside it is still "the one read-only web search tool and nothing else".
 */
export const PUBLIC_WEB_SEARCH_SERVER = "github-mcp-server";
export const PUBLIC_WEB_SEARCH_TOOL = "web_search";

function deny(feedback: string): PermissionRequestResult {
	return { kind: "reject", feedback };
}

function allow(): PermissionRequestResult {
	return { kind: "approve-once" };
}

export type GateOptions = {
	owner: string;
	repository: string;
	tools: Set<string>;
	active?: () => Promise<boolean>;
};

export function gate(options: GateOptions): PermissionHandler {
	return async (request: PermissionRequest): Promise<PermissionRequestResult> => {
		if (options.active && !(await options.active())) {
			return deny("The agent owner or repository permission is no longer active.");
		}
		if (request.kind === "custom-tool") {
			return options.tools.has(request.toolName)
				? allow()
				: deny(`${request.toolName} is not available to the planner.`);
		}
		return deny("The planner has no host filesystem, shell or URL access.");
	};
}

/** A worker may submit one terminal result and has no ambient capabilities. */
export function terminalGate(
	tool: string,
	active?: () => Promise<boolean>,
): PermissionHandler {
	return async (request: PermissionRequest): Promise<PermissionRequestResult> => {
		if (active && !(await active())) {
			return deny("The agent owner is no longer active.");
		}
		return request.kind === "custom-tool" && request.toolName === tool
			? allow()
			: deny("This worker may only submit its registered result.");
	};
}

/**
 * Public research receives no private tools and may only use hosted search.
 *
 * Search itself never reaches this gate: Anthropic runs it, bounded by the
 * session's `maxUses`, and what comes back is evidence rather than a call we
 * approve. `onWebSearchDenied` survives for the one thing that can still
 * refuse — a session whose owner went away mid-brief.
 */
export function publicResearchGate(
	resultTool: string,
	active?: () => Promise<boolean>,
	onWebSearchDenied?: () => void,
): PermissionHandler {
	return async (request: PermissionRequest): Promise<PermissionRequestResult> => {
		if (active && !(await active())) {
			onWebSearchDenied?.();
			return deny("The agent owner is no longer active.");
		}
		if (request.kind === "custom-tool") {
			return request.toolName === resultTool
				? allow()
				: deny("This worker may only submit its registered research result.");
		}
		if (request.kind === "url") return deny("Public research has no direct URL fetch capability.");
		return deny("Public research has no repository, filesystem, shell, or private document tools.");
	};
}
