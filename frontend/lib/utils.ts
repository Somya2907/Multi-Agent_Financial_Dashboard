import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatNumber(value: number | null | undefined, decimals = 2): string {
  if (value === null || value === undefined) return "N/A";
  return value.toFixed(decimals);
}

export function formatLargeNumber(value: number | null | undefined): string {
  if (value === null || value === undefined) return "N/A";
  if (Math.abs(value) >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
  if (Math.abs(value) >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
  if (Math.abs(value) >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
  return `$${value.toLocaleString()}`;
}

export function severityColor(severity: string) {
  switch (severity?.toUpperCase()) {
    case "HIGH":
      return { badge: "bg-red-950 text-red-300 border-red-800", dot: "bg-red-400", text: "text-red-400" };
    case "MEDIUM":
      return { badge: "bg-yellow-950 text-yellow-300 border-yellow-800", dot: "bg-yellow-400", text: "text-yellow-400" };
    case "LOW":
      return { badge: "bg-green-950 text-green-300 border-green-800", dot: "bg-green-400", text: "text-green-400" };
    default:
      return { badge: "bg-gray-800 text-gray-400 border-gray-700", dot: "bg-gray-400", text: "text-gray-400" };
  }
}

export function toneColor(tone: string) {
  switch (tone?.toLowerCase()) {
    case "positive":
      return "bg-green-950 text-green-300 border-green-800";
    case "negative":
      return "bg-red-950 text-red-300 border-red-800";
    case "mixed":
      return "bg-purple-950 text-purple-300 border-purple-800";
    default:
      return "bg-gray-800 text-gray-400 border-gray-700";
  }
}

export function confidenceColor(confidence: string) {
  switch (confidence?.toUpperCase()) {
    case "HIGH":
      return "text-green-400";
    case "MEDIUM":
      return "text-yellow-400";
    case "LOW":
      return "text-red-400";
    default:
      return "text-gray-400";
  }
}
