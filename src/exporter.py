import ezdxf
import os

def generate_dxf(constrained_primitives: list[dict], output_path: str):
    """
    Generates a DXF file from the constrained geometric primitives.
    
    Args:
        constrained_primitives: The final list of primitives with constraints.
        output_path: The file path to save the DXF file.
    """
    print(f"exporter.py: Generating DXF at {output_path}...")
    
    # --- ML Engineer 3 (UI/Integration Lead) implements the ezdxf logic here ---
    doc = ezdxf.new()
    msp = doc.modelspace()
    
    for prim in constrained_primitives:
        if prim['type'] == 'line':
            msp.add_line(prim['start'], prim['end'])
        elif prim['type'] == 'circle':
            msp.add_circle(prim['center'], prim['radius'])
            
    doc.saveas(output_path)
    print("DXF generation complete.")
    # -----------------------------------------------------------------------------
