import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { MutationCache, QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { toast } from "sonner";

import "./index.css";
import { App } from "./App";
import { errorMessage } from "./lib/api";
import { startEvents } from "./lib/events";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { refetchOnWindowFocus: false, retry: 1 },
  },
  // Every failed action tells the user why; callers add their own success toasts.
  mutationCache: new MutationCache({
    onError: (error) => {
      toast.error(errorMessage(error));
    },
  }),
});

startEvents(queryClient);

const root = document.getElementById("root");
if (root === null) throw new Error("missing #root");

createRoot(root).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
);
