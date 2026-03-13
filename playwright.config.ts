import { defineConfig, devices } from '@playwright/test';

const apiSuitePattern = /\bI\. Backend API\b/;

export default defineConfig({
    testDir: './tests/e2e',
    fullyParallel: false,
    forbidOnly: !!process.env.CI,
    retries: 0,
    workers: 1,
    reporter: [['html', { open: 'never' }], ['list']],
    timeout: 60000,

    use: {
        baseURL: 'http://localhost:5173',
        trace: 'on-first-retry',
        screenshot: 'on',
        video: 'off',
        actionTimeout: 15000,
    },

    projects: [
        {
            name: 'chromium',
            grepInvert: apiSuitePattern,
            use: { ...devices['Desktop Chrome'] },
        },
        {
            name: 'api',
            grep: apiSuitePattern,
        },
    ],
});
