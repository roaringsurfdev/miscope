import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';

// Local dev runs at http://localhost:4321/miscope/
export default defineConfig({
  site: 'https://roaringsurfdev.github.io',
  base: '/miscope',
  trailingSlash: 'always',
  integrations: [mdx()],
});
