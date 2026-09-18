import { describe, expect, it } from "bun:test";

import {
	assertWorkerTools,
	openWorker,
	plannerConfiguration,
	publicResearchConfiguration,
	workerConfiguration,
} from "./client";
import { gate, publicResearchGate, terminalGate } from "./permissions";
import { pullRequestTools } from "./pull-requests";
import { repositoryTools } from "./repository";

import type { PermissionRequest, Tool } from "./types";

describe("hosted agent configuration", () => {
	it("gives the planner exactly its own tools, every one of them gated", () => {
		let tool = {
			name: "read_plan",
			description: "read",
			parameters: {},
			handler: () => "ok",
		} as Tool;
		let config = plannerConfiguration(
			{ model: "model" },
			{ tools: [tool] },
			{
				token: "ghu_owner",
				repository: { id: "R_repo", owner: "octo-org", name: "score", defaultBranch: "main" },
				bootstrap: "Earlier in this room: nothing.",
			},
		);

		expect(config.model).toBe("model");
		expect(config.streaming).toBe(true);
		expect(config.tools).toHaveLength(1);
		expect(config.tools[0]?.skipPermission).toBe(false);
		expect(config.webSearch).toBeUndefined();
		expect(config.agent.prompt).toContain("read_repository_file");
		expect(config.agent.prompt).toContain("read_pull_request");
		expect(config.systemMessage).toContain("octo-org/score");
		expect(config.systemMessage).toContain("Earlier in this room: nothing.");
	});

	it("gives a worker only its terminal result tool", async () => {
		let result = {
			name: "submit_job_result",
			description: "submit",
			parameters: {},
			handler: () => "ok",
		} as Tool;
		let options = {
			token: "ghu_owner",
			name: "chopin-document-summary",
			prompt: "Summarize the supplied document and submit one result.",
			result,
			maxAiCredits: 32,
		};
		let config = workerConfiguration({ model: "model" }, options);
		expect(() => workerConfiguration({ model: "model" }, { ...options, maxAiCredits: 29 }))
			.toThrow("at least 30");

		expect(config.streaming).toBe(false);
		expect(config.maxRequests).toBe(16);
		expect(config.webSearch).toBeUndefined();
		expect(config.tools).toHaveLength(1);
		expect(config.tools[0]).toMatchObject({
			name: "submit_job_result",
			skipPermission: false,
			isTerminal: true,
		});
		expect(config.agent).toMatchObject({
			name: "chopin-document-summary",
			prompt: "Summarize the supplied document and submit one result.",
		});

		let decide = terminalGate("submit_job_result");
		expect(await decide({ kind: "custom-tool", toolName: "submit_job_result" }))
			.toEqual({ kind: "approve-once" });
		expect(await decide({ kind: "custom-tool", toolName: "read_plan" }))
			.toMatchObject({ kind: "reject" });
		expect(await decide({ kind: "url", url: "https://example.com" }))
			.toMatchObject({ kind: "reject" });
	});

	it("refuses a worker whose owner went away", async () => {
		let denials = 0;
		let decide = publicResearchGate("submit_research_result", async () => false, () => denials++);
		expect(await decide({ kind: "custom-tool", toolName: "submit_research_result" }))
			.toMatchObject({ kind: "reject" });
		expect(denials).toBe(1);
	});

	it("fails worker capability audits closed", () => {
		let result = { name: "submit_job_result", namespacedName: "custom:submit_job_result" };
		let web = {
			name: "web_search",
			mcpServerName: "github-mcp-server",
			mcpToolName: "web_search",
		};
		expect(() => assertWorkerTools([result], "submit_job_result")).not.toThrow();
		expect(() => assertWorkerTools([result, web], "submit_job_result", true)).not.toThrow();
		expect(() => assertWorkerTools([], "submit_job_result")).toThrow("received none");
		// The web tool is missing where it is required.
		expect(() => assertWorkerTools([result], "submit_job_result", true)).toThrow("mcp:web_search");
		// And present where it is not.
		expect(() => assertWorkerTools([result, web], "submit_job_result")).toThrow("web_search");
		expect(() =>
			assertWorkerTools(
				[result, { name: "web_fetch", namespacedName: "builtin:web_fetch" }],
				"submit_job_result",
			)
		).toThrow("builtin:web_fetch");
		expect(() =>
			assertWorkerTools([result, { ...web, mcpServerName: "ambient" }], "submit_job_result", true)
		).toThrow("mcp:web_search");
	});

	it("isolates public web research from private capabilities", async () => {
		let result = {
			name: "submit_research_result",
			description: "submit",
			parameters: {},
			handler: () => "ok",
		} as Tool;
		let config = publicResearchConfiguration({ model: "model" }, {
			token: "ghu_owner",
			name: "chopin-public-research",
			prompt: "Research only the disclosed public question.",
			result,
			maxAiCredits: 32,
		});
		expect(config.tools).toHaveLength(1);
		expect(config.tools[0]?.name).toBe("submit_research_result");
		expect(config.webSearch).toEqual({ maxUses: 12 });
		expect(config.streaming).toBe(false);

		let decide = publicResearchGate("submit_research_result");
		expect(await decide({ kind: "custom-tool", toolName: "submit_research_result" }))
			.toEqual({ kind: "approve-once" });
		expect(await decide({ kind: "custom-tool", toolName: "read_plan" }))
			.toMatchObject({ kind: "reject" });
		expect(await decide({ kind: "url", url: "https://example.com/evidence" }))
			.toMatchObject({ kind: "reject" });
		expect(
			await decide({
				kind: "mcp",
				serverName: "github-mcp-server",
				readOnly: true,
				toolName: "web_search",
			}),
		).toMatchObject({ kind: "reject" });
	});

	it("does not start a worker when the hosted agent is disabled", async () => {
		let result = {
			name: "submit_job_result",
			description: "submit",
			parameters: {},
			handler: () => "ok",
		} as Tool;
		await expect(openWorker(
			{ agent: false, model: "model", anthropicApiKey: "test-key" },
			{
				token: "ghu_owner",
				name: "chopin-document-summary",
				prompt: "Summarize the supplied document and submit one result.",
				result,
				maxAiCredits: 32,
			},
		)).rejects.toThrow("disabled");
	});

	it("denies everything the planner was not given", async () => {
		let decide = gate({
			owner: "octo-org",
			repository: "score",
			tools: new Set(["read_plan"]),
		});
		expect(await decide({ kind: "custom-tool", toolName: "read_plan" }))
			.toEqual({ kind: "approve-once" });
		expect(await decide({ kind: "custom-tool", toolName: "edit_repository" }))
			.toMatchObject({ kind: "reject" });
		expect(await decide({ kind: "shell", command: "cat /etc/passwd" }))
			.toMatchObject({ kind: "reject" });
		expect(await decide({ kind: "file", path: "/etc/passwd" }))
			.toMatchObject({ kind: "reject" });
		expect(await decide({ kind: "url", url: "https://example.com" }))
			.toMatchObject({ kind: "reject" });
		expect(
			await decide({
				kind: "mcp",
				serverName: "github",
				readOnly: true,
				toolName: "get_pull_request",
				args: { owner: "octo-org", repo: "score" },
			} as PermissionRequest),
		).toMatchObject({ kind: "reject" });
	});

	it("stops a planner whose owner lost the room", async () => {
		let decide = gate({
			owner: "octo-org",
			repository: "score",
			tools: new Set(["read_plan"]),
			active: async () => false,
		});
		expect(await decide({ kind: "custom-tool", toolName: "read_plan" }))
			.toMatchObject({ kind: "reject", feedback: expect.stringContaining("no longer active") });
	});
});

describe("hosted pull request tools", () => {
	it("binds every read to the channel's repository", async () => {
		let urls: URL[] = [];
		let tools = pullRequestTools({
			token: "ghu_owner",
			repository: { id: "R_repo", owner: "octo-org", name: "score", defaultBranch: "main" },
			fetch: async input => {
				let url = new URL(String(input));
				urls.push(url);
				if (url.pathname.endsWith("/files")) {
					return Response.json([{ filename: "src/a.ts", status: "modified", additions: 2 }]);
				}
				if (url.pathname.endsWith("/reviews")) {
					return Response.json([{ user: { login: "mona" }, state: "APPROVED" }]);
				}
				if (/\/pulls\/\d+$/.test(url.pathname)) {
					return Response.json({
						number: 7,
						title: "Change",
						state: "open",
						user: { login: "mona" },
						head: { ref: "topic" },
						base: { ref: "main" },
						additions: 2,
						deletions: 1,
					});
				}
				return Response.json([{ number: 7, title: "Change", state: "open", head: { ref: "t" } }]);
			},
		});
		let call = (name: string, input: unknown) => {
			let tool = tools.find(value => value.name === name)!;
			return (tool.handler as (raw: unknown) => Promise<string>)(input);
		};

		expect(await call("list_pull_requests", {})).toContain("Change");
		expect(await call("read_pull_request", { number: 7 })).toContain('"number": 7');
		expect(await call("list_pull_request_files", { number: 7 })).toContain("src/a.ts");
		expect(await call("list_pull_request_reviews", { number: 7 })).toContain("APPROVED");
		expect(await call("list_open_pull_requests_for_branch", { branch: "topic" })).toContain(
			"Change",
		);

		expect(urls.every(url => url.pathname.startsWith("/repos/octo-org/score/pulls"))).toBe(true);
		let branch = urls.at(-1)!;
		expect(branch.searchParams.get("head")).toBe("octo-org:topic");
	});

	it("refuses arguments it cannot trust and reports authorization loss", async () => {
		let token: string | undefined = "ghu_current";
		let tools = pullRequestTools({
			token: () => token,
			repository: { id: "R", owner: "o", name: "r", defaultBranch: "main" },
			fetch: async () => Response.json([]),
		});
		let call = (name: string, input: unknown) => {
			let tool = tools.find(value => value.name === name)!;
			return (tool.handler as (raw: unknown) => Promise<string>)(input);
		};

		expect(await call("read_pull_request", { number: 0 })).toContain(
			"positive pull request number",
		);
		expect(await call("list_pull_requests", { state: "merged" })).toContain("state must be");
		expect(await call("list_pull_requests", { limit: 500 })).toContain("limit must be");
		token = undefined;
		expect(await call("list_pull_requests", {})).toContain("authorization expired");
	});
});

describe("hosted repository tools", () => {
	it("binds every read to one repository and filters search results", async () => {
		let urls: URL[] = [];
		let tools = repositoryTools({
			token: "ghu_owner",
			repository: { id: "R_repo", owner: "octo-org", name: "score", defaultBranch: "main" },
			fetch: async input => {
				let url = new URL(String(input));
				urls.push(url);
				if (url.pathname.includes("/contents/")) {
					return Response.json({
						type: "file",
						encoding: "base64",
						content: Buffer.from("one\ntwo").toString("base64"),
					});
				}
				if (url.pathname.includes("/git/trees/")) {
					return Response.json({
						tree: [{ path: "src/a.ts", type: "blob", size: 10 }],
						truncated: false,
					});
				}
				if (url.pathname === "/search/code") {
					return Response.json({
						items: [
							{ path: "src/a.ts", html_url: "url", repository: { node_id: "R_repo" } },
							{ path: "secret", repository: { node_id: "R_other" } },
						],
					});
				}
				return Response.json([{
					sha: "abc",
					commit: { message: "change", author: { name: "Mona", date: "today" } },
				}]);
			},
		});
		let call = (name: string, input: unknown) => {
			let tool = tools.find(value => value.name === name)!;
			return (tool.handler as (raw: unknown) => Promise<string>)(input);
		};

		expect(await call("read_repository_file", { path: "src/a.ts" })).toContain("1: one");
		expect(await call("list_repository_tree", {})).toContain("src/a.ts");
		let searched = await call("search_repository", { terms: "symbol" });
		expect(searched).toContain("src/a.ts");
		expect(searched).not.toContain("secret");
		expect(await call("repository_history", {})).toContain("change");
		expect(
			urls.filter(url => url.pathname !== "/search/code").every(url =>
				url.pathname.startsWith("/repos/octo-org/score/")
			),
		).toBe(true);
		expect(urls.find(url => url.pathname === "/search/code")!.searchParams.get("q"))
			.toContain("repo:octo-org/score");
	});

	it("refuses paths that can escape the repository", async () => {
		let tools = repositoryTools({
			token: "token",
			repository: { id: "R", owner: "o", name: "r", defaultBranch: "main" },
			fetch: async () => Response.json({}),
		});
		let read = tools.find(tool => tool.name === "read_repository_file")!;
		let result = await (read.handler as (raw: unknown) => Promise<string>)({ path: "../secret" });
		expect(result).toContain("relative repository path");
	});

	it("resolves authorization again when a repository handler starts", async () => {
		let token: string | undefined = "ghu_current";
		let requests = 0;
		let tools = repositoryTools({
			token: () => token,
			repository: { id: "R", owner: "o", name: "r", defaultBranch: "main" },
			fetch: async (_input, init) => {
				requests++;
				expect(new Headers(init?.headers).get("authorization")).toBe("Bearer ghu_current");
				return Response.json({ tree: [], truncated: false });
			},
		});
		let tree = tools.find(tool => tool.name === "list_repository_tree")!;
		expect(await (tree.handler as (raw: unknown) => Promise<string>)({})).not.toContain("Error:");
		token = undefined;
		expect(await (tree.handler as (raw: unknown) => Promise<string>)({})).toContain(
			"authorization expired",
		);
		expect(requests).toBe(1);
	});
});
