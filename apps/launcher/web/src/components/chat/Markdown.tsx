import { memo, useDeferredValue, useMemo, type ReactNode } from "react";
import ReactMarkdown, { type Components, type Options } from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

import { CopyButton } from "@/components/CopyButton";
import { cn } from "@/lib/utils";
import { installKatexStyle } from "./katex-style";
import "./chat.css";

installKatexStyle();

const REMARK_PLUGINS: Options["remarkPlugins"] = [remarkGfm, remarkMath];
// KaTeX first: it replaces the `language-math` code blocks highlight.js would otherwise take.
const REHYPE_PLUGINS: Options["rehypePlugins"] = [[rehypeKatex, { strict: "ignore" }], rehypeHighlight];

/** The hast nodes react-markdown passes as `node`, as far as this module reads them. */
interface HastNode {
  type?: string;
  tagName?: string;
  value?: unknown;
  properties?: Record<string, unknown>;
  children?: HastNode[];
}

function nodeText(node: unknown): string {
  if (typeof node !== "object" || node === null) return "";
  const hast = node as HastNode;
  if (hast.type === "text" && typeof hast.value === "string") return hast.value;
  return Array.isArray(hast.children) ? hast.children.map(nodeText).join("") : "";
}

/** The `language-*` class of a <pre>'s <code>. */
function codeLanguage(node: unknown): string | null {
  if (typeof node !== "object" || node === null) return null;
  const code = (node as HastNode).children?.find((child) => child.type === "element" && child.tagName === "code");
  const classes = code?.properties?.className;
  const names = Array.isArray(classes) ? classes : typeof classes === "string" ? classes.split(/\s+/) : [];
  for (const name of names) {
    if (typeof name === "string" && name.startsWith("language-")) return name.slice("language-".length);
  }
  return null;
}

function CodeBlock({ node, children }: { node: unknown; children?: ReactNode }) {
  const language = codeLanguage(node);
  return (
    <div className="chat-code">
      <div className="chat-code-bar">
        <span className="truncate font-mono">{language ?? "text"}</span>
        <CopyButton text={nodeText(node).replace(/\n$/, "")} size="icon-xs" className="size-6" />
      </div>
      <pre>{children}</pre>
    </div>
  );
}

const COMPONENTS: Components = {
  pre: ({ node, children }) => <CodeBlock node={node}>{children}</CodeBlock>,
  a: ({ href, title, children }) => (
    <a href={href} title={title} target="_blank" rel="noreferrer noopener">
      {children}
    </a>
  ),
  // A model's words make the browser fetch nothing (an image's address can carry the chat to its
  // host): an image is a link to open.
  img: ({ src, alt, title }) => {
    const label = alt || (typeof src === "string" ? src : "") || "image";
    return typeof src === "string" && src !== "" ? (
      <a href={src} title={title} target="_blank" rel="noreferrer noopener">
        {label}
      </a>
    ) : (
      <span>{label}</span>
    );
  },
  table: ({ children }) => (
    <div className="chat-table">
      <table>{children}</table>
    </div>
  ),
};

// Code spans and fences (an unterminated fence runs to the end, as while it streams).
const CODE_SPANS = /(```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)|`[^`\n]+`)/g;

/** Models also write LaTeX as \[...\] and \(...\), which remark-math does not read: rewrite them
 * as $$...$$ and $...$ outside code. */
export function normalizeMath(text: string): string {
  if (!text.includes("\\[") && !text.includes("\\(")) return text;
  return text
    .split(CODE_SPANS)
    .map((part, index) =>
      index % 2 === 1
        ? part
        : part
            .replace(/\\\[([\s\S]+?)\\\]/g, (_match: string, math: string) => `\n$$\n${math.trim()}\n$$\n`)
            .replace(/\\\(([\s\S]+?)\\\)/g, (_match: string, math: string) => `$${math}$`),
    )
    .join("");
}

/** Markdown with GFM (tables, task lists), LaTeX and highlighted code. While text streams in,
 * rendering runs at a lower priority than input and skips versions it cannot keep up with. */
export const Markdown = memo(function Markdown({ text, className }: { text: string; className?: string }) {
  const deferred = useDeferredValue(text);
  const rendered = useMemo(
    () => (
      <ReactMarkdown remarkPlugins={REMARK_PLUGINS} rehypePlugins={REHYPE_PLUGINS} components={COMPONENTS}>
        {normalizeMath(deferred)}
      </ReactMarkdown>
    ),
    [deferred],
  );
  return <div className={cn("chat-md", className)}>{rendered}</div>;
});
