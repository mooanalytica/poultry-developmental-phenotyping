"""Minimal .xlsx writer using only the standard library (no spreadsheet package is installed here). One worksheet per DataFrame; numbers stored as numbers, text as inline strings."""

import re
import zipfile
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


def _col(i):
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def _sheet_xml(df):
    rows = []

    def cell(r, c, v):
        ref = f"{_col(c)}{r}"
        if v is None or (isinstance(v, float) and np.isnan(v)) or (v is pd.NA):
            return ""
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            return f'<c r="{ref}"><v>{repr(float(v))}</v></c>'
        t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(v))
        return f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{escape(t)}</t></is></c>'

    rows.append('<row r="1">' + "".join(cell(1, j, c) for j, c in enumerate(df.columns)) + "</row>")
    for i, rec in enumerate(df.itertuples(index=False), start=2):
        rows.append(f'<row r="{i}">' + "".join(cell(i, j, v) for j, v in enumerate(rec)) + "</row>")
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
        + "".join(rows)
        + "</sheetData></worksheet>"
    )


def write_xlsx(path, sheets):
    names = [re.sub(r"[\[\]\*\?/\\:]", "_", n)[:31] for n in sheets]

    class _Z(zipfile.ZipFile):  # fixed time stamps so identical content gives identical bytes
        def writestr(self, name, data, *a, **k):
            zi = zipfile.ZipInfo(name, date_time=(2025, 10, 17, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            return super().writestr(zi, data)

    with _Z(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            + "".join(
                f'<Override PartName="/xl/worksheets/sheet{i+1}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                for i in range(len(names))
            )
            + "</Types>",
        )
        z.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>',
        )
        z.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
            + "".join(
                f'<sheet name="{escape(n)}" sheetId="{i+1}" r:id="rId{i+1}"/>'
                for i, n in enumerate(names)
            )
            + "</sheets></workbook>",
        )
        z.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(
                f'<Relationship Id="rId{i+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i+1}.xml"/>'
                for i in range(len(names))
            )
            + "</Relationships>",
        )
        for i, df in enumerate(sheets.values()):
            z.writestr(f"xl/worksheets/sheet{i+1}.xml", _sheet_xml(df))
