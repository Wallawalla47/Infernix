import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/** Copies text, falling back to a hidden textarea where the async clipboard is unavailable
 * (plain http on a LAN address is not a secure context). */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // fall through to the textarea route
  }
  try {
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(area);
    return ok;
  } catch {
    return false;
  }
}

export function downloadText(fileName: string, text: string): void {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function readStorage(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function writeStorage(key: string, value: string | null): void {
  try {
    if (value === null) {
      window.localStorage.removeItem(key);
    } else {
      window.localStorage.setItem(key, value);
    }
  } catch {
    // storage unavailable (private window, blocked site data): the preference is per-visit only
  }
}

/** Path separator of a host path ("\\" for Windows drive or UNC paths). */
export function pathSeparator(path: string): "\\" | "/" {
  return /^[A-Za-z]:/.test(path) || path.startsWith("\\\\") || path.includes("\\") ? "\\" : "/";
}

export function joinPath(dir: string, name: string): string {
  const sep = pathSeparator(dir);
  if (dir.endsWith("\\") || dir.endsWith("/")) return dir + name;
  return dir + sep + name;
}

export function baseName(path: string): string {
  const parts = path.split(/[\\/]/).filter((part) => part.length > 0);
  return parts.length > 0 ? parts[parts.length - 1] : path;
}

/** "owner/name" from "owner/name", "https://huggingface.co/owner/name[/...]" or "hf.co/owner/name". */
export function parseRepo(input: string): string | null {
  const text = input.trim();
  if (text.length === 0) return null;
  const url = text.match(/^(?:https?:\/\/)?(?:www\.)?(?:huggingface\.co|hf\.co)\/([^/\s]+)\/([^/\s?#]+)/i);
  if (url) return `${url[1]}/${url[2]}`;
  const plain = text.match(/^([A-Za-z0-9][\w.-]*)\/([\w.-]+)$/);
  return plain ? `${plain[1]}/${plain[2]}` : null;
}

export function sum(values: number[]): number {
  let total = 0;
  for (const value of values) total += value;
  return total;
}
