import { Progress } from "@/components/ui/progress"

export default function Loading() {
  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="w-full max-w-md space-y-4 p-4">
        <h2 className="text-2xl font-bold text-center">Loading...</h2>
        <Progress value={33} className="w-full" />
        <p className="text-center text-muted-foreground">
          Preparing your video analysis experience
        </p>
      </div>
    </div>
  )
} 