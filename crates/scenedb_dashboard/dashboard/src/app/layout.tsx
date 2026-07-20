import type { Metadata } from "next";
import "./globals.css";
import LayoutClient from "./LayoutClient";

export const metadata: Metadata = {
  title: "SceneDB Dashboard",
  description: "Real-time SceneDB monitoring dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body>
        <LayoutClient>{children}</LayoutClient>
      </body>
    </html>
  );
}
