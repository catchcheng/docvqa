
from docling.document_converter import DocumentConverter



class docling_generator:
    def __init__(self):
        
        self.converter = DocumentConverter()
        
    
def docling(self, image_path):
    result = self.converter.convert(image_path)
    return result

