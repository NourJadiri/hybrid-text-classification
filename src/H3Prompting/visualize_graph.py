from graph import create_classification_graph
import os

# Create the classification graph
graph = create_classification_graph()

print("Generating graph visualization...")

try:
    # Try to save as PNG using local rendering
    from langchain_core.runnables.graph import MermaidDrawMethod
    
    # Save as PNG file using Pyppeteer (local rendering)
    png_data = graph.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.PYPPETEER,
        output_file_path="classification_graph.png"
    )
    print("✅ Graph saved as 'classification_graph.png'")
    
except Exception as e:
    print(f"❌ PNG generation failed: {e}")
    print("Trying to generate Mermaid text instead...")
    
    try:
        # Fallback: Generate Mermaid text
        mermaid_text = graph.get_graph().draw_mermaid()
        
        # Save as text file
        with open("classification_graph.mmd", "w") as f:
            f.write(mermaid_text)
        print("✅ Graph saved as Mermaid text in 'classification_graph.mmd'")
        print("You can copy this text to https://mermaid.live/ to visualize it")
        
    except Exception as e2:
        print(f"❌ Mermaid text generation failed: {e2}")
        print("Trying ASCII representation...")
        
        try:
            # Last fallback: ASCII representation
            ascii_graph = graph.get_graph().draw_ascii()
            
            # Save as text file
            with open("classification_graph_ascii.txt", "w") as f:
                f.write(ascii_graph)
            print("✅ Graph saved as ASCII representation in 'classification_graph_ascii.txt'")
            print("\nASCII Graph:")
            print(ascii_graph)
            
        except Exception as e3:
            print(f"❌ All visualization methods failed: {e3}")
            print("Manual graph structure:")
            print("START -> categories -> [narratives|handle_other_category]")
            print("narratives -> validate_narratives -> clean_narratives -> subnarratives -> clean_subnarratives -> write_results -> END")
            print("handle_other_category -> write_results -> END")