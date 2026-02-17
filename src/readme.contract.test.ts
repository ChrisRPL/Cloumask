import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

type PackageJson = {
	scripts?: Record<string, string>;
};

const root = process.cwd();
const readme = readFileSync(join(root, 'README.md'), 'utf8');
const pkg = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8')) as PackageJson;

describe('README contract', () => {
	it('includes key OSS contributor sections', () => {
		expect(readme).toContain('## Quickstart');
		expect(readme).toContain('## Architecture');
		expect(readme).toContain('## Run modes');
		expect(readme).toContain('## Verify');
		expect(readme).toContain('## Troubleshooting');
		expect(readme).toContain('## Contributing');
	});

	it('references only existing npm scripts', () => {
		const mentionedScripts = [...readme.matchAll(/npm run ([a-zA-Z0-9:_-]+)/g)].map((match) => match[1]);
		const uniqueScripts = [...new Set(mentionedScripts)];
		const scripts = pkg.scripts ?? {};
		const missing = uniqueScripts.filter((name) => !(name in scripts));

		expect(missing).toEqual([]);
	});
});
