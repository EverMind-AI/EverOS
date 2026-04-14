"use client";

import { cn } from "@/lib/utils";

interface FilterButtonProps {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}

export function FilterButton({ active, onClick, children }: FilterButtonProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex items-center text-sm px-4 py-1.5 rounded-full transition-colors cursor-pointer",
        active
          ? "bg-foreground text-background"
          : "bg-background text-foreground border border-border hover:bg-secondary"
      )}
    >
      {children}
    </button>
  );
}
