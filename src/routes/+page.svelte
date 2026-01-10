<script lang="ts">
	import { Button } from "$lib/components/ui/button";
	import * as Card from "$lib/components/ui/card";
	import { Input } from "$lib/components/ui/input";
	import { Badge } from "$lib/components/ui/badge";
	import { Separator } from "$lib/components/ui/separator";

	let greeting = $state("Welcome to Cloumask");
</script>

<main class="flex flex-col items-center justify-center min-h-screen p-8">
	<div class="text-center max-w-2xl">
		<!-- Logo wordmark -->
		<img
			src="/assets/icon_large.png"
			alt="Cloumask"
			class="h-16 mx-auto mb-8 object-contain"
		/>

		<h1 class="text-4xl font-bold text-foreground mb-4">{greeting}</h1>
		<p class="text-xl text-muted-foreground mb-2">
			Local-first AI for computer vision data processing
		</p>
		<p class="text-muted-foreground font-mono text-sm mb-8">
			<span class="code">From cloud to canvas</span>
		</p>

		<!-- shadcn Button variants -->
		<div class="flex gap-4 justify-center mb-12 flex-wrap">
			<Button>Get Started</Button>
			<Button variant="secondary">Learn More</Button>
			<Button variant="outline">Documentation</Button>
			<Button variant="ghost">Settings</Button>
		</div>

		<!-- Design system showcase with shadcn Card -->
		<Card.Root class="text-left">
			<Card.Header>
				<Card.Title>Design System Preview</Card.Title>
				<Card.Description>
					shadcn-svelte components with Cloumask theme
				</Card.Description>
			</Card.Header>
			<Card.Content class="space-y-6">
				<!-- Badge examples -->
				<div>
					<p class="text-sm text-muted-foreground mb-2">Badges</p>
					<div class="flex gap-2 flex-wrap">
						<Badge>Default</Badge>
						<Badge variant="secondary">Secondary</Badge>
						<Badge variant="outline">Outline</Badge>
						<Badge variant="destructive">Destructive</Badge>
					</div>
				</div>

				<Separator />

				<!-- Colors -->
				<div>
					<p class="text-sm text-muted-foreground mb-2">Brand Colors</p>
					<div class="flex gap-2">
						<div
							class="size-10 rounded-md bg-primary"
							title="Primary (Forest)"
						></div>
						<div
							class="size-10 rounded-md bg-background border border-border"
							title="Background (Cream)"
						></div>
						<div
							class="size-10 rounded-md bg-forest-light"
							title="Forest Light"
						></div>
						<div
							class="size-10 rounded-md bg-destructive"
							title="Destructive"
						></div>
					</div>
				</div>

				<Separator />

				<!-- Typography -->
				<div>
					<p class="text-sm text-muted-foreground mb-2">Typography</p>
					<p class="font-sans text-foreground">Inter - UI Text (sans-serif)</p>
					<p class="font-mono text-muted-foreground">
						JetBrains Mono - Code (monospace)
					</p>
				</div>

				<Separator />

				<!-- Input example with shadcn -->
				<div>
					<p class="text-sm text-muted-foreground mb-2">Input</p>
					<Input
						type="text"
						placeholder="Type something..."
						class="max-w-sm"
					/>
				</div>

				<Separator />

				<!-- Button sizes -->
				<div>
					<p class="text-sm text-muted-foreground mb-2">Button Sizes</p>
					<div class="flex gap-2 items-center flex-wrap">
						<Button size="sm">Small</Button>
						<Button size="default">Default</Button>
						<Button size="lg">Large</Button>
						<Button size="icon">+</Button>
					</div>
				</div>
			</Card.Content>
		</Card.Root>
	</div>
</main>
