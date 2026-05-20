import 'dotenv/config';
import { PostgresSaver } from '@langchain/langgraph-checkpoint-postgres';

async function main() {
  const checkpointer = PostgresSaver.fromConnString(process.env.DATABASE_URL!);
  await checkpointer.setup();
  console.log('LangGraph PostgresSaver tables created successfully.');
  process.exit(0);
}

main().catch((err) => {
  console.error('Checkpointer setup failed:', err);
  process.exit(1);
});
