// A configuration's environment variables (`env`): infernix-serve starts with them besides the
// launcher's own. Rows edit the draft; empty rows are dropped when it is saved.

import { Plus, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { EnvVar } from "@/lib/api-types";
import { envRowProblem } from "@/lib/configs";

export function EnvEditor({ env, onChange }: { env: EnvVar[]; onChange: (env: EnvVar[]) => void }) {
  const setRow = (index: number, row: EnvVar) => onChange(env.map((old, position) => (position === index ? row : old)));
  return (
    <div className="flex flex-col gap-2 py-2.5">
      {env.length === 0 ? (
        <div className="text-sm text-muted-foreground">
          None. Add a variable such as <code className="font-mono text-xs">CUDA_VISIBLE_DEVICES</code> for this
          configuration only.
        </div>
      ) : (
        <ul className="flex flex-col gap-2">
          {env.map((row, index) => {
            const problem = envRowProblem(env, index);
            return (
              <li key={index} className="flex flex-col gap-1">
                <div className="flex items-center gap-2">
                  <Input
                    value={row.name}
                    placeholder="NAME"
                    spellCheck={false}
                    autoComplete="off"
                    className="h-8 w-[38%] font-mono text-xs"
                    aria-label={`Variable ${index + 1} name`}
                    aria-invalid={problem !== null}
                    onChange={(event) => setRow(index, { ...row, name: event.target.value })}
                  />
                  <span className="text-muted-foreground">=</span>
                  <Input
                    value={row.value}
                    placeholder="value"
                    spellCheck={false}
                    autoComplete="off"
                    className="h-8 min-w-0 flex-1 font-mono text-xs"
                    aria-label={`Variable ${index + 1} value`}
                    onChange={(event) => setRow(index, { ...row, value: event.target.value })}
                  />
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    aria-label={`Remove ${row.name || `variable ${index + 1}`}`}
                    onClick={() => onChange(env.filter((_, position) => position !== index))}
                  >
                    <Trash2 />
                  </Button>
                </div>
                {problem ? <div className="text-xs text-critical">{problem}</div> : null}
              </li>
            );
          })}
        </ul>
      )}
      <Button variant="outline" size="sm" className="w-fit" onClick={() => onChange([...env, { name: "", value: "" }])}>
        <Plus />
        Add variable
      </Button>
    </div>
  );
}
