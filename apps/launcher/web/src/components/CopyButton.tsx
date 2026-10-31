import { useEffect, useState } from "react";
import { Check, Copy } from "lucide-react";
import { toast } from "sonner";

import { Button, type ButtonProps } from "@/components/ui/button";
import { copyText } from "@/lib/utils";

export function CopyButton({
  text,
  label,
  toastMessage,
  size = "icon-sm",
  variant = "ghost",
  className,
}: {
  text: string;
  /** Visible label; without it the button is an icon with an accessible name. */
  label?: string;
  toastMessage?: string;
  size?: ButtonProps["size"];
  variant?: ButtonProps["variant"];
  className?: string;
}) {
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (!copied) return;
    const id = window.setTimeout(() => setCopied(false), 1500);
    return () => window.clearTimeout(id);
  }, [copied]);

  const onClick = async () => {
    const ok = await copyText(text);
    if (ok) {
      setCopied(true);
      if (toastMessage) toast.success(toastMessage);
    } else {
      toast.error("Copying failed; select the text instead");
    }
  };

  return (
    <Button
      variant={variant}
      size={label ? (size === "icon-sm" ? "sm" : size) : size}
      className={className}
      onClick={() => void onClick()}
      aria-label={label ? undefined : "Copy"}
    >
      {copied ? <Check className="text-success" /> : <Copy />}
      {label ? <span>{copied ? "Copied" : label}</span> : null}
    </Button>
  );
}
