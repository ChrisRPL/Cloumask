<script lang="ts">
	let greeting = $state('Welcome to Cloumask');
</script>

<main class="flex flex-col items-center justify-center min-h-screen p-8">
	<div class="text-center max-w-2xl">
		<!-- Logo wordmark -->
		<img
			src="/assets/icon_large.png"
			alt="Cloumask"
			class="h-16 mx-auto mb-8 object-contain"
		/>

		<h1 class="text-4xl font-bold text-text-primary mb-4">{greeting}</h1>
		<p class="text-xl text-text-secondary mb-2">
			Local-first AI for computer vision data processing
		</p>
		<p class="text-text-muted font-mono text-sm mb-8">
			<span class="code">From cloud to canvas</span>
		</p>

		<div class="flex gap-4 justify-center mb-12">
			<button class="btn-primary cursor-pointer">Get Started</button>
			<button class="btn-secondary cursor-pointer">Learn More</button>
		</div>

		<!-- Design system showcase -->
		<div class="card p-6 text-left">
			<h2 class="text-lg font-semibold text-text-primary mb-4">Design System Preview</h2>

			<div class="space-y-4">
				<!-- Colors -->
				<div>
					<p class="text-sm text-text-muted mb-2">Brand Colors</p>
					<div class="flex gap-2">
						<div class="size-10 rounded-md bg-forest" title="Forest Green"></div>
						<div class="size-10 rounded-md bg-cream border border-border" title="Cream"></div>
						<div class="size-10 rounded-md bg-forest-light" title="Forest Light"></div>
						<div class="size-10 rounded-md bg-forest-dark" title="Forest Dark"></div>
					</div>
				</div>

				<!-- Typography -->
				<div>
					<p class="text-sm text-text-muted mb-2">Typography</p>
					<p class="font-sans text-text-primary">Inter - UI Text (sans-serif)</p>
					<p class="font-mono text-text-secondary">JetBrains Mono - Code (monospace)</p>
				</div>

				<!-- Input example -->
				<div>
					<p class="text-sm text-text-muted mb-2">Input</p>
					<input type="text" class="input-base w-full" placeholder="Type something..." />
				</div>
			</div>
		</div>
	</div>
</main>
