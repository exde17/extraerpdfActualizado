import fitz  # PyMuPDF
import tabula
import pandas as pd
import io
import json
import re
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
import os
from pdfminer.high_level import extract_text, extract_pages
from pdfminer.layout import LAParams, LTTextContainer, LTChar, LTLine, LTRect, LTFigure
from pdfminer.converter import TextConverter
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

def extract_tables_with_tabula(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract tables from PDF using tabula-py.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: List of dictionaries containing table data and page number
    """
    tables = []
    try:
        # Extract all tables from the PDF
        extracted_tables = tabula.read_pdf(
            pdf_path, 
            pages='all', 
            multiple_tables=True,
            lattice=True,  # Use lattice mode for tables with grid lines
            stream=True,   # Also try stream mode for tables without grid lines
            guess=True     # Enable table area guessing
        )
        
        current_page = 1
        for i, table in enumerate(extracted_tables):
            if not table.empty:
                # Format the table as a string with proper alignment
                table_str = format_pandas_table(table)
                
                # Add to our list
                tables.append({
                    "data": table,
                    "text": table_str,
                    "page": current_page,
                    "index": i
                })
            
            # Check if we need to move to the next page
            # This is an approximation since tabula doesn't return page numbers directly
            if i > 0 and i % 2 == 0:  # Assuming max 2 tables per page
                current_page += 1
    except Exception as e:
        print(f"Error extracting tables with tabula: {e}")
    
    return tables

def format_pandas_table(df: pd.DataFrame) -> str:
    """
    Format a pandas DataFrame as a nicely aligned text table similar to PDF24.
    
    Args:
        df (pd.DataFrame): The DataFrame to format
        
    Returns:
        str: Formatted table as text
    """
    # Replace NaN with empty string
    df = df.fillna('')
    
    # Get the maximum width for each column
    col_widths = {}
    for col in df.columns:
        # Get max width of column name and values
        col_widths[col] = max(
            len(str(col)),
            *[len(str(val)) for val in df[col]]
        )
    
    # Create header row
    header = " ".join(
        str(col).ljust(col_widths[col]) for col in df.columns
    )
    
    # Create data rows
    rows = []
    for _, row in df.iterrows():
        formatted_row = " ".join(
            str(row[col]).ljust(col_widths[col]) for col in df.columns
        )
        rows.append(formatted_row)
    
    # Combine all parts
    table_str = f"{header}\n\n"
    
    for row in rows:
        table_str += f"{row}\n"
    
    return table_str

def extract_text_with_pdfminer(pdf_path: str) -> str:
    """
    Extract text from PDF using PDFMiner with improved layout preservation.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text
    """
    output_string = io.StringIO()
    with open(pdf_path, 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(
            rsrcmgr, 
            output_string, 
            laparams=LAParams(
                line_margin=0.5,
                char_margin=2.0,
                word_margin=0.1,
                boxes_flow=0.5,
                detect_vertical=True
            )
        )
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)
    
    return output_string.getvalue()

def detect_and_format_tables_from_text(text: str) -> str:
    """
    Detect and format tables from extracted text based on spacing patterns.
    
    Args:
        text (str): Extracted text from PDF
        
    Returns:
        str: Text with tables formatted
    """
    lines = text.split('\n')
    formatted_lines = []
    tables_found = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this line could be a table header
        if is_potential_table_header(line):
            # Look ahead to see if we have a table
            table_lines = [line]
            j = i + 1
            
            # Skip empty lines after header
            while j < len(lines) and not lines[j].strip():
                j += 1
            
            # Collect potential table rows
            while j < len(lines) and is_potential_table_row(lines[j], line):
                table_lines.append(lines[j])
                j += 1
            
            # If we found enough rows, consider it a table
            if len(table_lines) >= 2:  
                # Format the table
                formatted_table = format_text_table(table_lines)
                formatted_lines.append(formatted_table)
                tables_found = True
                i = j
                continue
        
        formatted_lines.append(line)
        i += 1
    
    return '\n'.join(formatted_lines)

def is_potential_table_header(line: str) -> bool:
    """
    Check if a line could be a table header based on patterns.
    
    Args:
        line (str): Line to check
        
    Returns:
        bool: True if the line is likely a table header
    """
    # Check if the line has multiple words with significant spacing between them
    if not line.strip():
        return False
    
    # Check for header-like patterns
    header_patterns = [
        r'DESCRIPCION\s+Cant\s+W\s+Hrs',
        r'ELEMENTO\s+CANT\.\s+V\.UNIT\.',
        r'POT\.\s+ACTIVA\s+POT\.\s+REACTIVA',
        r'IMPUESTO\s+TOTAL\s+BASE',
        r'MEDIDOR\s+Serie\s+ID',
        r'ACTIVA\s+Lectura',
        r'Cant\s+W\s+Hrs',
        r'CANT\.\s+V\.UNIT\.\s+IVA',
        r'TOTAL\s+BASE\s+IMP\.'
    ]
    
    for pattern in header_patterns:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    
    # Check for multiple capital words separated by spaces
    words = line.split()
    if len(words) >= 2:  
        capital_words = sum(1 for word in words if word.isupper())
        if capital_words >= 2:
            return True
    
    # Check for multiple spaces as column separators
    spaces = re.findall(r'\s{2,}', line)
    return len(spaces) >= 1  

def is_potential_table_row(line: str, header: str) -> bool:
    """
    Check if a line could be a table row based on the header pattern.
    
    Args:
        line (str): Line to check
        header (str): The header line to compare with
        
    Returns:
        bool: True if the line is likely a table row
    """
    if not line.strip():
        return False
    
    # Check if the line has a similar spacing pattern to the header
    header_spaces = [match.start() for match in re.finditer(r'\s{2,}', header)]
    
    if not header_spaces:
        return False
    
    # Check if the line has content at similar positions
    for pos in header_spaces:
        if pos >= len(line):
            continue
        
        # Check if there's a space or content boundary near this position
        window = 8  
        for offset in range(-window, window + 1):
            check_pos = pos + offset
            if 0 <= check_pos < len(line) and line[check_pos].isspace():
                return True
    
    # Additional checks for table rows
    # Check if the line has numbers which is common in table data
    if re.search(r'\d+', line):
        return True
    
    # Check if the line has similar word count to the header
    header_words = len(header.split())
    line_words = len(line.split())
    if abs(header_words - line_words) <= 2:  
        return True
    
    return False

def format_text_table(table_lines: List[str]) -> str:
    """
    Format detected table lines to preserve alignment.
    
    Args:
        table_lines (List[str]): Lines that form a table
        
    Returns:
        str: Formatted table
    """
    # Ensure all lines have the same length for proper alignment
    max_length = max(len(line) for line in table_lines)
    padded_lines = [line.ljust(max_length) for line in table_lines]
    
    # Join the lines
    return '\n'.join(padded_lines)

def extract_text_with_layout(pdf_path: str) -> str:
    """
    Extract text from PDF while preserving layout using PyMuPDF.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text with preserved layout
    """
    doc = fitz.open(pdf_path)
    text_with_layout = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Extract text with layout information
        blocks = page.get_text("dict")["blocks"]
        
        # Sort blocks by y-coordinate (top to bottom)
        blocks.sort(key=lambda b: b["bbox"][1])
        
        page_text = []
        
        for block in blocks:
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                line_text = ""
                for span in line["spans"]:
                    line_text += span["text"]
                
                if line_text.strip():
                    page_text.append(line_text)
        
        text_with_layout.append("\n".join(page_text))
    
    doc.close()
    return "\n\n".join(text_with_layout)

def process_pdf_with_tables(pdf_path: str, output_txt_path: str, tables_json_path: str = None, output_csv_path: str = None):
    """
    Procesa un PDF extrayendo texto y tablas por separado
    
    Args:
        pdf_path (str): Ruta al archivo PDF
        output_txt_path (str): Ruta para guardar el texto extraído
        tables_json_path (str, optional): Ruta para guardar las tablas en formato JSON
        output_csv_path (str, optional): Ruta para guardar las tablas en formato CSV
    """
    # Extract text from PDF with layout preservation
    text_content = extract_text_with_layout(pdf_path)
    
    # Try to detect and format tables from the text
    text_with_formatted_tables = detect_and_format_tables_from_text(text_content)
    
    # Extract tables using tabula as a backup method
    tables = extract_tables_with_tabula(pdf_path)
    
    # Write to output file
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(text_with_formatted_tables)
    
    # Save tables to JSON if requested
    if tables_json_path and tables:
        tables_json = []
        for table in tables:
            tables_json.append({
                "page": table["page"],
                "index": table["index"],
                "data": table["data"].to_dict(orient="records")
            })
        
        with open(tables_json_path, "w", encoding="utf-8") as f:
            json.dump(tables_json, f, ensure_ascii=False, indent=2)
    
    # Save tables to CSV if requested
    if output_csv_path and tables:
        # Create a directory for CSV files
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        for i, table in enumerate(tables):
            csv_file = f"{output_csv_path}_table_{i+1}.csv"
            table["data"].to_csv(csv_file, index=False, encoding="utf-8")

def format_tables(tables: List[Dict[str, Any]]) -> str:
    """
    Formatea las tablas extraídas para su presentación
    
    Args:
        tables (list): Lista de tablas extraídas
        
    Returns:
        str: Texto formateado de las tablas
    """
    if not tables:
        return ""  
    
    formatted_text = ""
    
    for i, table in enumerate(tables):
        formatted_text += f"{table['text']}\n\n"
    
    return formatted_text

def process_pdf(pdf_path: str, output_txt_path: str, output_csv_path: str = None):
    """
    Procesa un PDF y extrae texto preservando la estructura lo mejor posible
    
    Args:
        pdf_path (str): Ruta al archivo PDF
        output_txt_path (str): Ruta para guardar el texto extraído
        output_csv_path (str, optional): Ruta para guardar las tablas en formato CSV
    """
    # Use process_pdf_with_tables for all processing to ensure tables are extracted
    process_pdf_with_tables(pdf_path, output_txt_path, output_csv_path=output_csv_path)

def extract_tables_from_pdf(pdf_path: str, output_csv_path: str) -> int:
    """
    Extrae tablas de un PDF usando tabula-py y las guarda en archivos CSV
    
    Args:
        pdf_path (str): Ruta al archivo PDF
        output_csv_path (str): Ruta base para guardar las tablas extraídas en formato CSV
        
    Returns:
        int: Número de tablas extraídas
    """
    tables = extract_tables_with_tabula(pdf_path)
    
    # Save tables to CSV
    if tables:
        # Create a directory for CSV files
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        for i, table in enumerate(tables):
            csv_file = f"{output_csv_path}_table_{i+1}.csv"
            table["data"].to_csv(csv_file, index=False, encoding="utf-8")
    
    return len(tables)

# Mantener las funciones de camelot para compatibilidad
def extract_tables_with_camelot(pdf_path):
    """
    Extrae tablas de un PDF usando camelot
    
    Args:
        pdf_path (str): Ruta al archivo PDF
        
    Returns:
        list: Lista de tablas extraídas con información de página
    """
    tables = []
    try:
        import camelot
        # Extract tables from the PDF
        extracted_tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
        
        for i, table in enumerate(extracted_tables):
            if not table.df.empty:
                tables.append({
                    "page": table.page,
                    "data": table.df,
                    "text": table.df.to_string(index=False),
                    "accuracy": table.accuracy,
                    "whitespace": table.whitespace
                })
    except Exception as e:
        print(f"Error extracting tables with camelot: {e}")
    
    return tables

def process_pdf_with_tables_camelot(pdf_path, output_txt_path, tables_json_path=None, output_csv_path=None):
    """
    Procesa un PDF extrayendo texto y tablas por separado
    
    Args:
        pdf_path (str): Ruta al archivo PDF
        output_txt_path (str): Ruta para guardar el texto extraído
        tables_json_path (str, optional): Ruta para guardar las tablas en formato JSON
        output_csv_path (str, optional): Ruta para guardar las tablas en formato CSV
    """
    # Extract text from PDF
    text_content = extract_text_with_layout(pdf_path)
    
    # Extract tables
    tables = extract_tables_with_camelot(pdf_path)
    
    # Format tables for text output
    formatted_tables = format_tables_camelot(tables)
    
    # Combine text and tables
    full_text = text_content
    
    # Write to output file
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    # Save tables to JSON if requested
    if tables_json_path and tables:
        tables_json = []
        for table in tables:
            tables_json.append({
                "page": table["page"],
                "data": table["data"].to_dict(orient="records"),
                "accuracy": table["accuracy"],
                "whitespace": table["whitespace"]
            })
        
        with open(tables_json_path, "w", encoding="utf-8") as f:
            json.dump(tables_json, f, ensure_ascii=False, indent=2)
    
    # Save tables to CSV if requested
    if output_csv_path and tables:
        for i, table in enumerate(tables):
            csv_file = f"{output_csv_path}_table_{i+1}.csv"
            table["data"].to_csv(csv_file, index=False, encoding="utf-8")
    
    return full_text, tables

def format_tables_camelot(tables):
    """
    Formatea las tablas extraídas para su presentación
    
    Args:
        tables (list): Lista de tablas extraídas
        
    Returns:
        str: Texto formateado de las tablas
    """
    if not tables:
        return ""  
    
    formatted_tables = []
    for i, table in enumerate(tables):
        formatted_tables.append(f"Tabla {i+1} (Página {table['page']}):\n\n{table['text']}")
    
    return "\n".join(formatted_tables)

def process_pdf_camelot(pdf_path, output_txt_path, output_csv_path=None):
    """
    Procesa un PDF y extrae texto preservando la estructura lo mejor posible
    
    Args:
        pdf_path (str): Ruta al archivo PDF
        output_txt_path (str): Ruta para guardar el texto extraído
        output_csv_path (str, optional): Ruta para guardar las tablas en formato CSV
    """
    # Extraer texto y tablas
    full_text, tables = process_pdf_with_tables_camelot(pdf_path, output_txt_path, output_csv_path=output_csv_path)
    
    # Formatear las tablas
    formatted_tables = format_tables_camelot(tables)
    
    # Combinar texto y tablas
    combined_content = full_text + "\n\n" + formatted_tables
    
    # Guardar el contenido combinado
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(combined_content)
    
    return {
        "text": full_text,
        "tables": tables,
        "tables_count": len(tables)
    }

def extract_tables_from_pdf_camelot(pdf_path, output_csv_path):
    """
    Extrae tablas de un PDF usando Camelot y las guarda en un archivo CSV
    
    Args:
        pdf_path (str): Ruta al archivo PDF
        output_csv_path (str): Ruta para guardar las tablas extraídas en formato CSV
        
    Returns:
        int: Número de tablas extraídas
    """
    tables = extract_tables_with_camelot(pdf_path)
    
    if tables:
        for i, table in enumerate(tables):
            csv_file = f"{output_csv_path}_table_{i+1}.csv"
            table["data"].to_csv(csv_file, index=False, encoding="utf-8")
    
    return len(tables)
