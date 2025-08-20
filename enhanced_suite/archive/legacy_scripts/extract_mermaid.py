import re

def extract_mermaid_blocks():
    with open('RADIODIFF_MERGED_STANDARDIZED_REPORT.md', 'r') as f:
        content = f.read()
    
    # Split content and find mermaid sections
    lines = content.split('\n')
    mermaid_blocks = []
    current_block = []
    in_mermaid = False
    
    for line in lines:
        if '```mermaid' in line:
            in_mermaid = True
            current_block = []
        elif '```' in line and in_mermaid:
            in_mermaid = False
            if current_block:
                mermaid_blocks.append('\n'.join(current_block))
        elif in_mermaid:
            current_block.append(line)
    
    print(f'Found {len(mermaid_blocks)} mermaid blocks')
    
    # Save unique blocks
    unique_blocks = []
    for i, block in enumerate(mermaid_blocks):
        if block not in unique_blocks:
            unique_blocks.append(block)
            with open(f'radiodiff_standardized_report_mermaid/diagram_{len(unique_blocks)}.mmd', 'w') as f:
                f.write(block)
            print(f'Saved diagram_{len(unique_blocks)}.mmd')
    
    print(f'Total unique diagrams: {len(unique_blocks)}')
    
    # Show first block for verification
    if unique_blocks:
        print('\nFirst block:')
        print(unique_blocks[0][:200] + '...' if len(unique_blocks[0]) > 200 else unique_blocks[0])

if __name__ == '__main__':
    extract_mermaid_blocks()