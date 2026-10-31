// The full-page sign-in card shown while the launcher wants an admin key from this browser (a
// management call or the event stream answered 401 sign_in), and the header's sign-out menu.

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { KeyRound, LogIn, LogOut, ShieldCheck } from "lucide-react";
import { toast } from "sonner";

import { Logo } from "@/components/Logo";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { SESSION_KEY, useSession } from "@/lib/admin";
import { ApiFailure, api, errorMessage } from "@/lib/api";
import type { Session } from "@/lib/api-types";
import { requireSignIn, signedIn } from "@/lib/events";

function signInError(error: unknown): string {
  if (error instanceof ApiFailure && (error.status === 401 || error.status === 403)) {
    return error.message.length > 0 ? error.message : "That key was not accepted. Use an admin key.";
  }
  return errorMessage(error);
}

export function SignInPage() {
  const queryClient = useQueryClient();
  const [key, setKey] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Called directly rather than through useMutation: a refused key is answered here, not by the
  // global error toast.
  const submit = async () => {
    const value = key.trim();
    if (value.length === 0 || pending) return;
    setPending(true);
    setError(null);
    try {
      const session = await api.signIn(value);
      queryClient.setQueryData<Session>(SESSION_KEY, session);
      setKey("");
      toast.success(session.key_name ? `Signed in with ${session.key_name}` : "Signed in");
      signedIn(queryClient);
    } catch (failure) {
      setError(signInError(failure));
    } finally {
      setPending(false);
    }
  };

  return (
    <div className="flex min-h-dvh items-center justify-center bg-background px-4 py-10">
      <Card className="hero-glow w-full max-w-md gap-5 py-6">
        <CardHeader className="gap-3 px-6">
          <Logo />
          <div className="flex flex-col gap-1">
            <CardTitle className="text-base">
              <KeyRound className="size-4 text-muted-foreground" />
              Sign in to manage this launcher
            </CardTitle>
            <CardDescription className="text-sm leading-relaxed">
              This page is open from another computer, so managing the launcher needs an admin API key. Create one under
              Settings, API keys, on the computer running infernix-launcher.
            </CardDescription>
          </div>
        </CardHeader>
        <CardContent className="px-6">
          <form
            className="flex flex-col gap-3"
            onSubmit={(event) => {
              event.preventDefault();
              void submit();
            }}
          >
            <Input
              type="password"
              value={key}
              autoFocus
              autoComplete="off"
              spellCheck={false}
              placeholder="inx-…"
              aria-label="Admin API key"
              aria-invalid={error !== null}
              className="font-mono"
              onChange={(event) => setKey(event.target.value)}
            />
            {error ? (
              <Alert variant="destructive" className="py-2">
                <AlertDescription className="text-foreground">{error}</AlertDescription>
              </Alert>
            ) : null}
            <Button type="submit" disabled={pending || key.trim().length === 0}>
              <LogIn />
              {pending ? "Signing in…" : "Sign in"}
            </Button>
            <p className="flex items-start gap-1.5 text-xs leading-relaxed text-muted-foreground">
              <ShieldCheck className="mt-0.5 size-3.5 shrink-0" />
              The launcher keeps the sign-in in a cookie for this browser. API keys (not admin) only reach the
              OpenAI/Anthropic endpoints.
            </p>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}

/** "Signed in as ..." with Sign out, in the header; nothing while no sign-in is in use. */
export function SessionMenu() {
  const queryClient = useQueryClient();
  const { data: session } = useSession();
  const [pending, setPending] = useState(false);
  if (!session?.signed_in) return null;

  const signOut = async () => {
    setPending(true);
    try {
      const next = await api.signOut();
      queryClient.setQueryData<Session>(SESSION_KEY, next);
      toast.info("Signed out");
      if (next.required && !next.signed_in) requireSignIn();
    } catch (failure) {
      toast.error(errorMessage(failure));
    } finally {
      setPending(false);
    }
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon-sm" aria-label="Session">
          <KeyRound />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuLabel>Signed in{session.key_name ? ` with ${session.key_name}` : ""}</DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem disabled={pending} onSelect={() => void signOut()}>
          <LogOut />
          Sign out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
