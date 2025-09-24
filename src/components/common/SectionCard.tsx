import React from "react";

import { Card } from "./Card";

type SectionCardProps = {
  title: string;
  right?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
};

export function SectionCard({ title, right, children, className }: SectionCardProps) {
  return (
    <Card title={title} right={right} className={className}>
      {children}
    </Card>
  );
}
