import asyncio
from dotenv import load_dotenv
from agents.extensions.sandbox import (
    DaytonaSandboxClient,
    DaytonaSandboxClientOptions,
)

load_dotenv()


async def main():
    sandbox_client = DaytonaSandboxClient()

    sandbox = await sandbox_client.create(options=DaytonaSandboxClientOptions())

    async with sandbox:
        result = await sandbox.exec("pwd && echo hello > test.txt && ls -la")

        print("Exit code:", result.exit_code)
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)

        stream = await sandbox.read("test.txt")
        print(stream.read().decode())
        stream.close()


if __name__ == "__main__":
    asyncio.run(main())
