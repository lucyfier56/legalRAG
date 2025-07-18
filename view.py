from qdrant_client import QdrantClient

def export_qdrant_data_to_txt(
    output_file="qdrant_data1_export.txt",
    qdrant_url="http://localhost:6333",
    collection_name="test_documents"
):
    client = QdrantClient(url=qdrant_url)
    offset = None
    batch_size = 100
    total = 0

    with open(output_file, "w", encoding="utf-8") as f:
        while True:
            response = client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            points = response.points if hasattr(response, "points") else response[0]
            for point in points:
                payload = point.payload
                text = payload.get("text") or payload.get("content") or "<no text>"
                # Write an ID header and text snippet to file
                f.write(f"ID: {point.id}\n")
                f.write(text + "\n")
                f.write("-" * 80 + "\n\n")
                total += 1

            offset = getattr(response, "next_page_offset", None) or (response[1] if isinstance(response, tuple) else None)
            if not offset:
                break

    print(f"Export completed. Total points saved: {total}")
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    export_qdrant_data_to_txt()