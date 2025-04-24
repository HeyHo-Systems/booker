from PIL import Image, ImageDraw, ImageFont
import os

# Create a new image with white background
width = 800
height = 1000
image = Image.new('RGB', (width, height), 'white')
draw = ImageDraw.Draw(image)

# Add invoice content
text_color = (0, 0, 0)  # Black
y_position = 50

# Invoice content
invoice_content = [
    "INVOICE",
    "",
    "Invoice Date: 2024-03-15",
    "Invoice #: INV-2024-001",
    "",
    "Bill To:",
    "John Doe",
    "123 Main Street",
    "Anytown, ST 12345",
    "",
    "Payment Method: VISA",
    "",
    "Description                Amount",
    "------------------------   -------",
    "Web Development Services   $750.00",
    "Hosting (Monthly)         $50.00",
    "",
    "Total Amount:             $800.00 USD",
]

# Draw each line
for line in invoice_content:
    draw.text((50, y_position), line, fill=text_color)
    y_position += 30

# Save the image
output_path = os.path.join('test_invoices', 'web_services_invoice.png')
image.save(output_path)
print(f"Created test invoice: {output_path}") 