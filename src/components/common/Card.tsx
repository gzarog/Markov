import React from "react";

type CardProps = {
  title: string;
  right?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
};

export function Card({ title, right, children, className }: CardProps) {
  return (
    <section className={`bg-white rounded-2xl shadow ${className || ""}`}>
      <header className="px-4 py-3 flex items-center justify-between border-b">
        <h3 className="text-base md:text-lg font-semibold">{title}</h3>
        <div className="text-xs text-gray-500">{right}</div>
      </header>
      <div className="p-3 md:p-4">{children}</div>
    </section>
  );
}
