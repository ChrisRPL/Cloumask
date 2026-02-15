const API_BASE_URL = 'http://localhost:8765';
const WINDOWS_ABSOLUTE_PATH = /^[A-Za-z]:[\\/]/;
const URL_SCHEME = /^[a-zA-Z][a-zA-Z\d+\-.]*:/;

/**
 * Convert local file paths into backend-served URLs for browser mode.
 * Leaves data/blob/http URLs untouched.
 */
export function toLocalImageUrl(pathOrUrl: string): string {
	if (!pathOrUrl) return pathOrUrl;
	if (pathOrUrl.startsWith('data:') || pathOrUrl.startsWith('blob:')) {
		return pathOrUrl;
	}
	if (
		pathOrUrl.startsWith('/api/') ||
		pathOrUrl.startsWith('/@fs/') ||
		pathOrUrl.startsWith('/assets/') ||
		pathOrUrl.startsWith('/_app/')
	) {
		return pathOrUrl;
	}
	if (pathOrUrl.startsWith('/')) {
		return `${API_BASE_URL}/api/review/image?path=${encodeURIComponent(pathOrUrl)}`;
	}
	if (WINDOWS_ABSOLUTE_PATH.test(pathOrUrl)) {
		return `${API_BASE_URL}/api/review/image?path=${encodeURIComponent(pathOrUrl)}`;
	}
	if (URL_SCHEME.test(pathOrUrl)) {
		return pathOrUrl;
	}
	return `${API_BASE_URL}/api/review/image?path=${encodeURIComponent(pathOrUrl)}`;
}
