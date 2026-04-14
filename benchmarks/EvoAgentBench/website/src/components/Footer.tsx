export function Footer() {
  return (
    <footer className="border-t border-border py-8 mt-auto">
      <div className="mx-auto max-w-6xl px-6 text-center text-sm text-muted-foreground">
        <p>EvoAgentBench &copy; {new Date().getFullYear()}. Built for the research community.</p>
      </div>
    </footer>
  );
}
