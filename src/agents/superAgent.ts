import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { Runnable } from "@langchain/core/runnables";
import { z } from "zod";
import { llm } from "../utils";

async function createSuperAgent(
  systemPrompt: string,
  members: string[]
): Promise<Runnable> {
  const options = ["FINISH", ...members] as const;

  const routeTool = {
    name: "route",
    description: "Select the next role.",
    schema: z.object({
      reasoning: z
        .string()
        .describe("Explain the decision behind selecting the next role."),
      next: z.enum(options),
      instructions: z
        .string()
        .describe(
          "The specific instructions of the sub-task the next role should be accomplish."
        ),
    }),
  };

  // Enhance the system prompt with explicit instructions to prevent loops
  const enhancedSystemPrompt = `${systemPrompt}

IMPORTANT WORKFLOW GUIDELINES:
1. Each task should progress toward completion with each step
2. Avoid calling the same agent multiple times in sequence
3. If the requested task is complete or no further progress can be made, select FINISH
4. Maximum workflow steps should not exceed 10 before FINISH is selected
5. Research → Processing → Chart Generation → Email is the typical workflow
`;

  let prompt = ChatPromptTemplate.fromMessages([
    ["system", enhancedSystemPrompt],
    new MessagesPlaceholder("messages"),
    [
      "system",
      "Given the conversation above, who should act next? Or should we FINISH? Select one of {options}. Remember to FINISH once the task is complete or after 10 steps maximum.",
    ],
  ]);

  prompt = await prompt.partial({
    options: options.join(", "),
    team_members: members.join(", "),
  });

  // Add state tracking to prevent loops
  const visitCount = new Map<string, number>();

  let supervisor = prompt
    .pipe(
      llm.bindTools([routeTool], {
        runName: "route",
      })
    )
    .pipe((x) => {
      if (x.tool_calls && x.tool_calls.length > 0) {
        const nextAgent = x.tool_calls[0].args.next;
        
        // Increment visit count
        const count = (visitCount.get(nextAgent) || 0) + 1;
        visitCount.set(nextAgent, count);
        
        // Force FINISH if we've called the same agent too many times
        if (nextAgent !== "FINISH" && count > 3) {
          console.log(`Forcing FINISH: ${nextAgent} called too many times (${count})`);
          return {
            reasoning: "Task is being terminated to prevent infinite loops.",
            next: "FINISH",
            instructions: "",
          };
        }
        
        return {
          reasoning: x.tool_calls[0].args.reasoning,
          next: nextAgent,
          instructions: x.tool_calls[0].args.instructions,
        };
      }
      
      return {
        next: options[0],
        instructions: "",
      };
    });

  return supervisor;
}

export default createSuperAgent;