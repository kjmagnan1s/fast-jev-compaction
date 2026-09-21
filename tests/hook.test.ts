import { describe, expect, it, vi } from 'vitest';
import {
  compactSession,
  decisionLog,
  decisionLogLines,
  register,
  resolveHookConfig,
  shouldPrune,
  summarize,
  toSessionMessages,
  triggerTokens,
} from '../hooks/fast-jev.ts';
import { applyDecisions, collectToolCalls, decideCall, type Message } from '../src/index.js';

type SessionMessage = Message & { handle?: string };

function message(role: Message['role'], text: string, extra: Partial<SessionMessage> = {}): SessionMessage {
  return { role, text, toolUses: [], ...extra };
}

function call(id: string, tool: string, input: Record<string, unknown>, text: string): SessionMessage {
  return message('assistant', '', {
    toolUses: [{ tool_use_id: id, tool, input, text }],
    handle: `h-${id}`,
  });
}

function result(id: string, text: string, isError = false): SessionMessage {
  return message('user', '', { toolResults: [{ tool_use_id: id, text, isError }], handle: `r-${id}` });
}

const fileA = 'export const a = 1;\n'.repeat(50);

function transcript(): SessionMessage[] {
  return [
    message('user', 'Fix the failing test.', { handle: 'h-0' }),
    call('tool-1', 'Read', { file_path: 'src/a.ts' }, fileA),
    result('tool-1', fileA),
    call('tool-2', 'Bash', { command: 'npm test' }, 'FAIL'),
    result('tool-2', 'FAIL b.test.ts: expected 2 to be 3', true),
    message('assistant', 'Fixing now.', { handle: 'h-5' }),
    message('user', 'go ahead', { handle: 'h-6' }),
  ];
}

function jevFetch(answer: (name: string) => number, bodies: string[] = []) {
  return async (_url: string, init?: { body?: string }) => {
    bodies.push(init?.body ?? '');
    const { questions } = JSON.parse(init?.body ?? '{}') as { questions: Record<string, unknown> };
    const answers = Object.fromEntries(
      Object.keys(questions).map((key) => [key, { type: 'noul', noul: answer(key) }]),
    );
    return { status: 200, ok: true, text: JSON.stringify({ answers }) };
  };
}

describe('hook config', () => {
  it('reads userConfig values and falls back to defaults', () => {
    expect(resolveHookConfig({})).toEqual({
      compactAtTokens: 250_000,
      compactAtPercent: 60,
      retryAfterTokens: 50_000,
      minReductionRatio: 0.25,
      model: 'jev-latest',
    });
    expect(
      resolveHookConfig({ apiKey: 'k', keepThreshold: 0.3, maxStateTokens: 1000, model: 'jev-x', goal: 'g', compactAtPercent: 'no' }),
    ).toEqual({
      apiKey: 'k',
      keepThreshold: 0.3,
      maxStateTokens: 1000,
      model: 'jev-x',
      goal: 'g',
      compactAtTokens: 250_000,
      compactAtPercent: 60,
      retryAfterTokens: 50_000,
      minReductionRatio: 0.25,
    });
  });
});

describe('session message mapping', () => {
  it('returns the engine objects for untouched messages and handle-less copies for rebuilt ones', () => {
    const messages = transcript();
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 }),
      decideCall(calls[1]!, { keepCall: 0.9, keepResult: 0.9 }, { keepThreshold: 0.5 }),
    ];
    messages[1]!.toolUses[0]!.text = 'x'.repeat(2000);
    messages[2]!.toolResults![0]!.text = 'x'.repeat(2000);
    const out = toSessionMessages(messages, applyDecisions(messages, decisions, calls, 300));
    expect(out).toHaveLength(messages.length);
    expect(out[0]).toBe(messages[0]);
    expect(out[1]?.handle).toBeUndefined();
    expect(out[1]?.toolUses[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[fast-jev-compaction truncated 1700 chars`),
    );
    expect(out[2]?.handle).toBeUndefined();
    expect(out[2]?.toolResults?.[0]?.text).toMatch(
      new RegExp(`^${'x'.repeat(300)}\\n\\[fast-jev-compaction truncated 1700 chars`),
    );
    expect(out[2]?.toolResults?.[0]).toMatchObject({ tool_use_id: 'tool-1', isError: false });
    expect(out[3]).toBe(messages[3]);
    expect(out[4]).toBe(messages[4]);
  });

  it('preserves short dropped-result messages and their handles', () => {
    const messages = transcript();
    messages[1]!.toolUses[0]!.text = 'y'.repeat(100);
    messages[2]!.toolResults![0]!.text = 'y'.repeat(100);
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 }),
      decideCall(calls[1]!, { keepCall: 0.9, keepResult: 0.9 }, { keepThreshold: 0.5 }),
    ];
    const out = toSessionMessages(messages, applyDecisions(messages, decisions, calls, 300));
    expect(out[1]).toBe(messages[1]);
    expect(out[2]).toBe(messages[2]);
  });
});

describe('compactSession', () => {
  it('runs the library over the engine fetch and reports the outcome', async () => {
    const bodies: string[] = [];
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k', model: 'jev-x' };
    const { result: output, messages } = await compactSession(
      transcript(),
      config,
      jevFetch((name) => (name === 'call_t2' || name === 'result_t2' ? 0.9 : 0.1), bodies),
    );
    expect(bodies).toHaveLength(1);
    expect(JSON.parse(bodies[0]!).model).toBe('jev-x');
    expect(output.decisions.map((d) => d.action)).toEqual(['drop_call', 'keep']);
    expect(messages.map((m) => m.handle)).toEqual(['h-0', 'h-tool-2', 'r-tool-2', 'h-5', 'h-6']);
    expect(summarize(output)).toMatch(/^\d+% reduction; 1 kept, 1 call_dropped; state ~\d+ tokens \(full\) in 1 request\(s\)$/);
    expect(decisionLog(output)).toBe('t1:Read:drop_call/call=0.10/result=0.10 t2:Bash:keep/call=0.90/result=0.90');
    expect(decisionLogLines(output)).toEqual([`decisions: ${decisionLog(output)}`]);
  });

  it('splits a long decision log into ui.log lines under the host limit', async () => {
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k' };
    const { result: output } = await compactSession(transcript(), config, jevFetch(() => 0.1));
    const lines = decisionLogLines(output, 60);
    expect(lines).toEqual([
      'decisions (1/2): t1:Read:drop_call/call=0.10/result=0.10',
      'decisions (2/2): t2:Bash:drop_call/call=0.10/result=0.10',
    ]);
    expect(lines.every((line) => line.length <= 60)).toBe(true);
    expect(decisionLogLines({ ...output, decisions: [] })).toEqual(['decisions: (none)']);
  });

  it('throws on a missing key and on failed requests so the hook falls back', async () => {
    const config = resolveHookConfig({ preserveRecentMessages: 1 });
    await expect(compactSession(transcript(), config, jevFetch(() => 0))).rejects.toThrow(/TYPESAFE_API_KEY/);
    await expect(
      compactSession(transcript(), { ...config, apiKey: 'k' }, async () => ({ status: 500, ok: false, text: 'x' })),
    ).rejects.toThrow(/500/);
  });
});

type Handler = (...args: never[]) => Promise<unknown>;
type HostFetch = ReturnType<typeof jevFetch>;

function registered(options: Record<string, unknown> = {}): Record<string, Handler> {
  const handlers: Record<string, Handler> = {};
  const on = (name: string, handler: Handler) => {
    handlers[name] = handler;
  };
  register(on as never, options as never);
  return handlers;
}

function host(fetch: HostFetch, context = { tokens: 0, window: 1_000_000 }) {
  const logs: string[] = [];
  let compactResult: unknown = { messages: [] };
  const compact = vi.fn(async () => compactResult);
  const $ = {
    env: { get: async () => 'k' },
    settings: { read: async () => ({}) },
    ui: { log: (text: string) => logs.push(text), toast: (text: string) => logs.push(text) },
    http: { fetch },
    session: { usage: async () => ({ context }), compact },
  };
  return {
    $,
    logs,
    compact,
    context,
    compactReturns(result: unknown) {
      compactResult = result;
    },
  };
}

async function runCompact(
  fetch: HostFetch,
  event: Record<string, unknown>,
): Promise<{ result: unknown; next: ReturnType<typeof vi.fn>; logs: string[] }> {
  const handlers = registered({ preserveRecentMessages: 1 });
  const { $, logs } = host(fetch);
  const next = vi.fn(async () => ({ messages: 'built-in summary' }));
  const result = await handlers['session.compact']!(
    $ as never,
    { messages: transcript(), ...event } as never,
    next as never,
  );
  return { result, next, logs };
}

describe('session.compact never summarizes the main conversation', () => {
  it('returns the pruned history when Jev clears the minimum', async () => {
    const { result, next } = await runCompact(jevFetch(() => 0.1), { trigger: 'plugin' });
    expect(next).not.toHaveBeenCalled();
    expect((result as { messages: unknown[] }).messages).toHaveLength(3);
  });

  it('skips instead of summarizing when nothing can be pruned', async () => {
    const { result, next, logs } = await runCompact(jevFetch(() => 0.9), { trigger: 'plugin' });
    expect(next).not.toHaveBeenCalled();
    expect(result).toEqual({ skip: expect.stringMatching(/^nothing to prune/) });
    expect(logs.at(-1)).toMatch(/history left as is, no summary .*run \/handoff when ready$/);
  });

  it('skips a small automatic prune but applies it on a typed /compact', async () => {
    const dropT2 = () => jevFetch((name) => (name.endsWith('t2') ? 0.1 : 0.9));
    const auto = await runCompact(dropT2(), { trigger: 'plugin' });
    expect(auto.result).toEqual({ skip: expect.stringMatching(/^below 25% minimum/) });
    const manual = await runCompact(dropT2(), { trigger: 'manual' });
    expect(manual.next).not.toHaveBeenCalled();
    expect((manual.result as { messages: unknown[] }).messages).toHaveLength(5);
  });

  it('skips when Jev fails', async () => {
    const failing = (async () => ({ status: 500, ok: false, text: 'x' })) as HostFetch;
    const { result, next } = await runCompact(failing, { trigger: 'manual' });
    expect(next).not.toHaveBeenCalled();
    expect(result).toEqual({ skip: expect.stringMatching(/500/) });
  });

  it('keeps the built-in fallback for subagent transcripts', async () => {
    const failing = (async () => ({ status: 500, ok: false, text: 'x' })) as HostFetch;
    const { result, next } = await runCompact(failing, { trigger: 'auto', agentId: 'agent-1' });
    expect(next).toHaveBeenCalledOnce();
    expect(result).toEqual({ messages: 'built-in summary' });
  });

  it('never runs on precompute', async () => {
    const bodies: string[] = [];
    const { result, next } = await runCompact(jevFetch(() => 0.1, bodies), { trigger: 'precompute' });
    expect(bodies).toHaveLength(0);
    expect(next).not.toHaveBeenCalled();
    expect(result).toEqual({ skip: expect.any(String) });
  });

  it('sends the text typed after /compact to Jev as the goal', async () => {
    const bodies: string[] = [];
    await runCompact(jevFetch(() => 0.1, bodies), {
      trigger: 'manual',
      instructions: ' keep the migration plan ',
    });
    expect(JSON.parse(bodies[0]!).state.goal).toBe('keep the migration plan');
  });
});

describe('turn.complete trigger', () => {
  it('uses the token trigger, or the percentage of the window when it is 0', () => {
    expect(triggerTokens({ compactAtTokens: 250_000, compactAtPercent: 60 }, 1_000_000)).toBe(250_000);
    expect(triggerTokens({ compactAtTokens: 0, compactAtPercent: 25 }, 200_000)).toBe(50_000);
  });

  it('waits for retryAfterTokens of growth after a skip', () => {
    expect(shouldPrune(249_999, 250_000, undefined, 50_000)).toBe(false);
    expect(shouldPrune(250_000, 250_000, undefined, 50_000)).toBe(true);
    expect(shouldPrune(290_000, 250_000, 260_000, 50_000)).toBe(false);
    expect(shouldPrune(310_000, 250_000, 260_000, 50_000)).toBe(true);
  });

  it('prunes above the trigger, backs off after a skip, and resets below it', async () => {
    const handlers = registered();
    const h = host(jevFetch(() => 0.1));
    const turn = async (tokens: number) => {
      h.context.tokens = tokens;
      await handlers['turn.complete']!(h.$ as never, {} as never, (async (e: unknown) => e) as never);
    };
    await turn(200_000);
    expect(h.compact).not.toHaveBeenCalled();
    h.compactReturns({ skip: 'nothing to prune' });
    await turn(260_000);
    expect(h.compact).toHaveBeenCalledTimes(1);
    await turn(290_000);
    expect(h.compact).toHaveBeenCalledTimes(1);
    await turn(310_000);
    expect(h.compact).toHaveBeenCalledTimes(2);
    await turn(100_000);
    h.compactReturns({ messages: [] });
    await turn(255_000);
    expect(h.compact).toHaveBeenCalledTimes(3);
    await turn(256_000);
    expect(h.compact).toHaveBeenCalledTimes(4);
  });
});
