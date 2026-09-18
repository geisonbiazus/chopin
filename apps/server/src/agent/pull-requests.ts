/**
 * Pull requests, read directly.
 *
 * These were the hosted GitHub MCP server's `pull_requests` toolset, reached
 * over HTTP with the user's token forwarded to it. They are ordinary REST
 * reads now, alongside the repository tools they sit next to, for the same
 * reason those are: the token stays in this process, and the repository is
 * closed over rather than passed in.
 *
 * That last part is the boundary. The MCP gate had to inspect arguments for a
 * foreign owner or repo and refuse calls that named somebody else's code;
 * there is no argument to inspect here, because no tool takes one.
 */

import type { Tool } from "./types";
import type { HostedRepository } from "./repository";

type Options = {
	token: string | (() => string | undefined);
	repository: HostedRepository;
	fetch?: (input: string | URL | Request, init?: RequestInit) => Promise<Response>;
};

const API = "https://api.github.com";
const TIMEOUT_MS = 15_000;
const MAX_RESPONSE_BYTES = 5 * 1_024 * 1_024;
const MAX_DIFF_BYTES = 256 * 1_024;
const MAX_PAGE = 50;

function object(value: unknown): Record<string, unknown> | undefined {
	return value && typeof value === "object" && !Array.isArray(value)
		? value as Record<string, unknown>
		: undefined;
}

function number(value: unknown): number {
	if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 1) {
		throw new Error("number must be a positive pull request number");
	}
	return value;
}

function bounded(value: unknown, fallback: number): number {
	if (value === undefined) return fallback;
	if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 1 || value > MAX_PAGE) {
		throw new Error(`limit must be an integer between 1 and ${MAX_PAGE}`);
	}
	return value;
}

function state(value: unknown): "open" | "closed" | "all" {
	if (value === undefined) return "open";
	if (value !== "open" && value !== "closed" && value !== "all") {
		throw new Error("state must be open, closed or all");
	}
	return value;
}

async function answer(work: () => Promise<unknown>): Promise<string> {
	try {
		return JSON.stringify(await work(), null, 2);
	} catch (err) {
		return `Error: ${err instanceof Error ? err.message : String(err)}`;
	}
}

/** The fields worth spending context on; a pull request payload is enormous. */
function summarize(value: Record<string, unknown>): Record<string, unknown> {
	let user = object(value.user);
	let head = object(value.head);
	let base = object(value.base);
	return {
		number: value.number,
		title: value.title,
		state: value.draft ? "draft" : value.state,
		author: user?.login,
		createdAt: value.created_at,
		updatedAt: value.updated_at,
		mergedAt: value.merged_at,
		head: head?.ref,
		base: base?.ref,
	};
}

export function pullRequestTools(options: Options): Tool[] {
	let call = async (
		path: string,
		init: { search?: URLSearchParams; diff?: boolean } = {},
	): Promise<unknown> => {
		let token = typeof options.token === "string" ? options.token : options.token();
		if (!token) throw new Error("GitHub authorization expired");
		let url = new URL(path, API);
		if (init.search) url.search = init.search.toString();
		let response = await (options.fetch ?? fetch)(url, {
			headers: {
				accept: init.diff ? "application/vnd.github.v3.diff" : "application/vnd.github+json",
				authorization: `Bearer ${token}`,
				"user-agent": "chopin",
				"x-github-api-version": "2022-11-28",
			},
			redirect: "error",
			signal: AbortSignal.timeout(TIMEOUT_MS),
		});
		if (!response.ok) throw new Error(`GitHub pull request read failed (${response.status})`);
		let source = await response.text();
		if (Buffer.byteLength(source) > MAX_RESPONSE_BYTES) {
			throw new Error("GitHub response is too large");
		}
		if (init.diff) {
			return Buffer.byteLength(source) > MAX_DIFF_BYTES
				? `${source.slice(0, MAX_DIFF_BYTES)}\n… diff truncated at 256 KiB.`
				: source;
		}
		try {
			return JSON.parse(source);
		} catch {
			throw new Error("GitHub returned an unreadable response");
		}
	};

	let root = `/repos/${encodeURIComponent(options.repository.owner)}/${
		encodeURIComponent(options.repository.name)
	}`;
	return [
		{
			name: "list_pull_requests",
			description: "List pull requests in the selected repository, most recently updated first.",
			parameters: {
				type: "object",
				properties: {
					state: { type: "string", enum: ["open", "closed", "all"] },
					limit: { type: "integer", minimum: 1, maximum: MAX_PAGE },
				},
				additionalProperties: false,
			},
			handler: raw =>
				answer(async () => {
					let input = object(raw) ?? {};
					let search = new URLSearchParams({
						state: state(input.state),
						sort: "updated",
						direction: "desc",
						per_page: String(bounded(input.limit, 20)),
					});
					let value = await call(`${root}/pulls`, { search });
					if (!Array.isArray(value)) throw new Error("GitHub did not return a pull request list");
					return value.map(item => summarize(object(item) ?? {}));
				}),
		},
		{
			name: "read_pull_request",
			description:
				"Read one pull request in the selected repository: its description, state and change size.",
			parameters: {
				type: "object",
				properties: { number: { type: "integer", minimum: 1 } },
				required: ["number"],
				additionalProperties: false,
			},
			handler: raw =>
				answer(async () => {
					let input = object(raw) ?? {};
					let value = object(await call(`${root}/pulls/${number(input.number)}`));
					if (!value) throw new Error("GitHub did not return a pull request");
					return {
						...summarize(value),
						body: value.body,
						merged: value.merged,
						mergeable: value.mergeable,
						changedFiles: value.changed_files,
						additions: value.additions,
						deletions: value.deletions,
						commits: value.commits,
					};
				}),
		},
		{
			name: "list_pull_request_files",
			description: "List the files one pull request in the selected repository changes.",
			parameters: {
				type: "object",
				properties: {
					number: { type: "integer", minimum: 1 },
					limit: { type: "integer", minimum: 1, maximum: MAX_PAGE },
				},
				required: ["number"],
				additionalProperties: false,
			},
			handler: raw =>
				answer(async () => {
					let input = object(raw) ?? {};
					let search = new URLSearchParams({ per_page: String(bounded(input.limit, 50)) });
					let value = await call(`${root}/pulls/${number(input.number)}/files`, { search });
					if (!Array.isArray(value)) throw new Error("GitHub did not return a file list");
					return value.map(item => {
						let file = object(item) ?? {};
						return {
							path: file.filename,
							status: file.status,
							additions: file.additions,
							deletions: file.deletions,
							previousPath: file.previous_filename,
						};
					});
				}),
		},
		{
			name: "read_pull_request_diff",
			description:
				"Read the unified diff of one pull request in the selected repository, truncated at 256 KiB.",
			parameters: {
				type: "object",
				properties: { number: { type: "integer", minimum: 1 } },
				required: ["number"],
				additionalProperties: false,
			},
			handler: raw =>
				answer(async () => {
					let input = object(raw) ?? {};
					return {
						number: input.number,
						diff: await call(`${root}/pulls/${number(input.number)}`, { diff: true }),
					};
				}),
		},
		{
			name: "list_pull_request_reviews",
			description:
				"List submitted reviews on one pull request in the selected repository, with their verdicts.",
			parameters: {
				type: "object",
				properties: {
					number: { type: "integer", minimum: 1 },
					limit: { type: "integer", minimum: 1, maximum: MAX_PAGE },
				},
				required: ["number"],
				additionalProperties: false,
			},
			handler: raw =>
				answer(async () => {
					let input = object(raw) ?? {};
					let search = new URLSearchParams({ per_page: String(bounded(input.limit, 20)) });
					let value = await call(`${root}/pulls/${number(input.number)}/reviews`, { search });
					if (!Array.isArray(value)) throw new Error("GitHub did not return a review list");
					return value.map(item => {
						let review = object(item) ?? {};
						return {
							author: object(review.user)?.login,
							state: review.state,
							submittedAt: review.submitted_at,
							body: review.body,
						};
					});
				}),
		},
		{
			name: "list_open_pull_requests_for_branch",
			description:
				"Find open pull requests in the selected repository whose head is the named branch.",
			parameters: {
				type: "object",
				properties: { branch: { type: "string", minLength: 1, maxLength: 255 } },
				required: ["branch"],
				additionalProperties: false,
			},
			handler: raw =>
				answer(async () => {
					let input = object(raw) ?? {};
					if (typeof input.branch !== "string" || !input.branch.trim()) {
						throw new Error("branch must be a non-empty string");
					}
					// Qualified with the owner because GitHub's head filter is
					// owner:ref, and an unqualified ref matches forks too.
					let search = new URLSearchParams({
						state: "open",
						head: `${options.repository.owner}:${input.branch.trim()}`,
					});
					let value = await call(`${root}/pulls`, { search });
					if (!Array.isArray(value)) throw new Error("GitHub did not return a pull request list");
					return value.map(item => summarize(object(item) ?? {}));
				}),
		},
	] satisfies Tool[] as Tool[];
}
