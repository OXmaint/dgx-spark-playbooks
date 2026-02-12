import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

SERVERS = {
    "rag-wo-server": {
        "command": "python",
        "args": ["tools/mcp_servers/rag_wo.py"],
        "transport": "stdio",
    }
}

async def maybe_call(obj, name, *args, **kwargs):
    fn = getattr(obj, name, None)
    if callable(fn):
        return await fn(*args, **kwargs)
    return None

async def main():
    client = MultiServerMCPClient(SERVERS)

    # Old/new API compatibility: some versions don't need start()/stop()
    await maybe_call(client, "start")

    # (Not required, but useful to validate server is up)
    try:
        tools = await client.get_tools()
        print("Tools available:", [t.name for t in tools])
    except Exception as e:
        print("Warning: get_tools failed (continuing):", e)

    # 1) Route to DOCS via org name (auto-resolves oceanix vs oceanix_docs)
    r1 = await client.call_tool(
        server_name="rag-wo-server",
        tool_name="route_and_search",
        arguments={"query": "gear box lubrication schedule", "organization_name": "oceanix", "k": 5},
    )
    print("\n=== route_and_search → docs ===")
    print(r1)

    # 2) Route to WORK ORDERS via org id
    r2 = await client.call_tool(
        server_name="rag-wo-server",
        tool_name="route_and_search",
        arguments={"query": "burnt compressor leads", "organization_id": "OCE-001", "k": 5},
    )
    print("\n=== route_and_search → work orders ===")
    print(r2)

    # Optional direct tools:
    d = await client.call_tool(
        server_name="rag-wo-server",
        tool_name="search_documents",
        arguments={"collection": "oceanix", "query": "lubrication schedule", "k": 5},
    )
    print("\n=== search_documents ===")
    print(d)

    w = await client.call_tool(
        server_name="rag-wo-server",
        tool_name="search_work_orders",
        arguments={"organization_id": "OCE-001", "query": "compressor leads", "k": 5},
    )
    print("\n=== search_work_orders ===")
    print(w)

    await maybe_call(client, "stop")

if __name__ == "__main__":
    asyncio.run(main())
