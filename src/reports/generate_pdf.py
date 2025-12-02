# src/reports/generate_pdf.py
import io
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
except ImportError:
    raise ImportError("Установите reportlab: pip install reportlab")

try:
    import plotly.graph_objects as go  # нужно для fig
except ImportError:
    pass

import pandas as pd
import numpy as np


def generate_pdf_report(grid: np.ndarray, df: pd.DataFrame, threshold: float, processed: int, total: int, fig):
    """
    Генерирует красивый PDF-отчёт
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1.5*cm, bottomMargin=1.5*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    elements = []
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Center', alignment=1, fontSize=14, spaceAfter=20))
    styles.add(ParagraphStyle(name='TitleBig', fontSize=20, alignment=1, spaceAfter=30, textColor=colors.darkred, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name='Subtitle', fontSize=12, alignment=1, spaceAfter=10, textColor=colors.grey))

    # === Заголовок ===
    elements.append(Paragraph("NDR-СИСТЕМА: ОТЧЁТ О ВЫЯВЛЕННЫХ УГРОЗАХ", styles['TitleBig']))
    elements.append(Paragraph(f"Сгенерировано: {datetime.now().strftime('%d.%m.%Y в %H:%M:%S')}", styles['Center']))
    elements.append(Spacer(1, 15))

    # === График пятна ===
    elements.append(Paragraph("Тепловая карта угроз (16×16)", styles['Heading2']))
    img_data = io.BytesIO()
    fig.write_image(img_data, format="png", width=720, height=720, scale=1.5)
    img_data.seek(0)
    img = Image(img_data, width=16*cm, height=16*cm)
    img.hAlign = 'CENTER'
    elements.append(img)
    elements.append(Spacer(1, 20))

    # === Статистика ===
    elements.append(Paragraph("Статистика обнаружения", styles['Heading2']))
    critical = int((grid >= threshold).sum())
    high = int((grid >= 0.8).sum())
    medium = int((grid >= 0.7).sum())
    suspicious = int((grid >= 0.5).sum())

    stats_data = [
        ["Показатель", "Значение"],
        ["Критические зоны", f"{critical:,}"],
        ["Высокая угроза (≥0.8)", f"{high:,}"],
        ["Аномалии (≥0.7)", f"{medium:,}"],
        ["Подозрительно (≥0.5)", f"{suspicious:,}"],
        ["Обработано потоков", f"{processed:,} из {total:,}"],
        ["Порог обнаружения", f"{threshold:.4f}"],
        ["Процент обработки", f"{processed/total*100:.1f}%"],
    ]
    table = Table(stats_data, colWidths=[9*cm, 6*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ALIGN', (1,1), (-1,-1), 'RIGHT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))

    # === Топ-10 зон ===
    elements.append(Paragraph("ТОП-10 самых опасных зон", styles['Heading2']))
    flat = [(x, y, grid[x,y]) for x in range(16) for y in range(16)]
    top10 = sorted(flat, key=lambda x: x[2], reverse=True)[:10]
    top_data = [["№", "X", "Y", "Угроза"]]
    for i, (x, y, score) in enumerate(top10, 1):
        status = "Критично" if score >= threshold else "Высокая"
        top_data.append([i, x+1, y+1, f"{score:.4f} ({status})"])
    top_table = Table(top_data)
    top_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.darkred),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.8, colors.red),
    ]))
    elements.append(top_table)

    # === Подвал ===
    elements.append(PageBreak())
    elements.append(Paragraph("Отчёт сгенерирован автоматически гибридной NDR-системой", styles['Subtitle']))
    elements.append(Paragraph("XGBoost + Isolation Forest | Дипломный проект", styles['Subtitle']))

    # === Генерация ===
    doc.build(elements)
    buffer.seek(0)
    return buffer