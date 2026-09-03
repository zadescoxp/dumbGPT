import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "dumbGPT",
  description: "A small local language model chat.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="dark h-full antialiased">
      <body className="flex h-full min-h-full flex-col overflow-hidden">{children}</body>
    </html>
  );
}
