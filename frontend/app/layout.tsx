import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Capitol Hill · Vision Zero Risk Map',
  description: 'Intersection crash-risk model for Capitol Hill, Seattle — NB + Empirical-Bayes pipeline.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
