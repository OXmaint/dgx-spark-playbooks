#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Report Generator for batch image analysis results.

Generates reports in various formats (Markdown, HTML) from batch analysis results.
Each report includes images with their analyses in a professional format.
"""

import base64
from datetime import datetime
from typing import List, Dict, Any, Optional

from logger import logger
from postgres_storage import PostgreSQLConversationStorage


class ReportGenerator:
    """Generates reports from batch image analysis results.

    Supports multiple output formats:
    - Markdown: Clean markdown with embedded base64 images
    - HTML: Styled HTML document with embedded images
    """

    def __init__(self, postgres_storage: PostgreSQLConversationStorage):
        """Initialize the report generator.

        Args:
            postgres_storage: PostgreSQL storage for retrieving results and images
        """
        self.storage = postgres_storage

    async def generate_report(
        self,
        batch_id: str,
        format: str = "markdown",
        include_images: bool = True
    ) -> str:
        """Generate a report for a completed batch analysis.

        Args:
            batch_id: Batch identifier
            format: Output format (markdown, html)
            include_images: Whether to embed images in the report

        Returns:
            Report content as string
        """
        # Get batch job details
        batch_job = await self.storage.get_batch_job(batch_id)
        if not batch_job:
            raise ValueError(f"Batch job not found: {batch_id}")

        if batch_job["status"] != "completed":
            raise ValueError(f"Batch job is not completed. Status: {batch_job['status']}")

        # Get all results
        results = await self.storage.get_batch_results(batch_id)

        if format == "html":
            return await self._generate_html_report(batch_job, results, include_images)
        else:
            return await self._generate_markdown_report(batch_job, results, include_images)

    async def _generate_markdown_report(
        self,
        batch_job: Dict[str, Any],
        results: List[Dict[str, Any]],
        include_images: bool
    ) -> str:
        """Generate a Markdown format report.

        Args:
            batch_job: Batch job details
            results: List of analysis results
            include_images: Whether to embed images

        Returns:
            Markdown report content
        """
        lines = []

        # Header
        lines.append("# Image Analysis Report")
        lines.append("")
        lines.append(f"**Batch ID:** {batch_job['batch_id']}")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Total Images:** {batch_job['total_images']}")
        lines.append(f"**Analysis Prompt:** {batch_job['analysis_prompt']}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Summary
        successful = sum(1 for r in results if not r.get("error_message"))
        failed = sum(1 for r in results if r.get("error_message"))

        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Successfully Analyzed:** {successful}")
        lines.append(f"- **Failed:** {failed}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Individual results
        lines.append("## Analysis Results")
        lines.append("")

        for i, result in enumerate(results, 1):
            lines.append(f"### {i}. {result['image_filename']}")
            lines.append("")

            if result.get("error_message"):
                lines.append(f"**Error:** {result['error_message']}")
                lines.append("")
                continue

            # Embed image if requested
            if include_images:
                image_data = await self.storage.get_image(result["image_id"])
                if image_data:
                    # Check if it's already a data URL
                    if image_data.startswith("data:"):
                        lines.append(f"![{result['image_filename']}]({image_data})")
                    else:
                        # Assume JPEG if no content type info
                        lines.append(f"![{result['image_filename']}](data:image/jpeg;base64,{image_data})")
                    lines.append("")

            # Analysis content
            lines.append("#### Analysis")
            lines.append("")
            lines.append(result["analysis_result"])
            lines.append("")
            lines.append(f"*Processed at: {result['processed_at']}*")
            lines.append("")
            lines.append("---")
            lines.append("")

        # Footer
        lines.append("")
        lines.append("---")
        lines.append("*Report generated by Batch Image Analysis System*")

        return "\n".join(lines)

    async def _generate_html_report(
        self,
        batch_job: Dict[str, Any],
        results: List[Dict[str, Any]],
        include_images: bool
    ) -> str:
        """Generate an HTML format report.

        Args:
            batch_job: Batch job details
            results: List of analysis results
            include_images: Whether to embed images

        Returns:
            HTML report content
        """
        # Summary stats
        successful = sum(1 for r in results if not r.get("error_message"))
        failed = sum(1 for r in results if r.get("error_message"))

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Analysis Report - {batch_job['batch_id'][:8]}</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .report-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .report-header h1 {{
            margin: 0 0 20px 0;
        }}
        .meta-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }}
        .meta-item {{
            background: rgba(255,255,255,0.2);
            padding: 10px;
            border-radius: 5px;
        }}
        .meta-label {{
            font-size: 0.8em;
            opacity: 0.8;
        }}
        .meta-value {{
            font-weight: bold;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary h2 {{
            margin-top: 0;
            color: #333;
        }}
        .stats {{
            display: flex;
            gap: 20px;
        }}
        .stat {{
            padding: 15px 25px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat.success {{
            background: #d4edda;
            color: #155724;
        }}
        .stat.error {{
            background: #f8d7da;
            color: #721c24;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
        }}
        .stat-label {{
            font-size: 0.9em;
        }}
        .result-card {{
            background: white;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .result-header {{
            background: #f8f9fa;
            padding: 15px 20px;
            border-bottom: 1px solid #eee;
        }}
        .result-header h3 {{
            margin: 0;
            color: #333;
        }}
        .result-body {{
            padding: 20px;
        }}
        .result-image {{
            text-align: center;
            margin-bottom: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .result-image img {{
            max-width: 100%;
            max-height: 500px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        .analysis-content {{
            white-space: pre-wrap;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .analysis-content h1, .analysis-content h2, .analysis-content h3,
        .analysis-content h4, .analysis-content h5 {{
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        .analysis-content h1:first-child, .analysis-content h2:first-child {{
            margin-top: 0;
        }}
        .analysis-content ul, .analysis-content ol {{
            margin: 10px 0;
            padding-left: 25px;
        }}
        .analysis-content li {{
            margin: 5px 0;
        }}
        .analysis-content table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .analysis-content table th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            border: 1px solid #5568d3;
        }}
        .analysis-content table td {{
            padding: 10px 12px;
            border: 1px solid #dee2e6;
        }}
        .analysis-content table tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        .analysis-content table tr:hover {{
            background: #e9ecef;
        }}
        .error-message {{
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.85em;
            margin-top: 15px;
            font-style: italic;
        }}
        .footer {{
            text-align: center;
            color: #666;
            padding: 20px;
            margin-top: 20px;
        }}
        .prompt-box {{
            background: #e7f3ff;
            border: 1px solid #b6d4fe;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
        }}
        .prompt-label {{
            font-weight: bold;
            color: #0a58ca;
            margin-bottom: 5px;
        }}
    </style>
</head>
<body>
    <div class="report-header">
        <h1>Image Analysis Report</h1>
        <div class="meta-info">
            <div class="meta-item">
                <div class="meta-label">Batch ID</div>
                <div class="meta-value">{batch_job['batch_id'][:8]}...</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Generated</div>
                <div class="meta-value">{datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Total Images</div>
                <div class="meta-value">{batch_job['total_images']}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Completed At</div>
                <div class="meta-value">{batch_job['completed_at'] or 'N/A'}</div>
            </div>
        </div>
        <div class="prompt-box">
            <div class="prompt-label">Analysis Prompt</div>
            <div>{batch_job['analysis_prompt']}</div>
        </div>
    </div>

    <div class="summary">
        <h2>Summary</h2>
        <div class="stats">
            <div class="stat success">
                <div class="stat-number">{successful}</div>
                <div class="stat-label">Successfully Analyzed</div>
            </div>
            <div class="stat error">
                <div class="stat-number">{failed}</div>
                <div class="stat-label">Failed</div>
            </div>
        </div>
    </div>
"""

        # Add individual results
        for i, result in enumerate(results, 1):
            html += f"""
    <div class="result-card">
        <div class="result-header">
            <h3>{i}. {result['image_filename']}</h3>
        </div>
        <div class="result-body">
"""
            if result.get("error_message"):
                html += f"""
            <div class="error-message">
                <strong>Error:</strong> {result['error_message']}
            </div>
"""
            else:
                # Add image if requested
                if include_images:
                    image_data = await self.storage.get_image(result["image_id"])
                    if image_data:
                        if image_data.startswith("data:"):
                            img_src = image_data
                        else:
                            img_src = f"data:image/jpeg;base64,{image_data}"
                        html += f"""
            <div class="result-image">
                <img src="{img_src}" alt="{result['image_filename']}">
            </div>
"""

                # Convert markdown-style content to basic HTML
                analysis_html = self._markdown_to_html(result["analysis_result"])
                html += f"""
            <div class="analysis-content">
                {analysis_html}
            </div>
            <div class="timestamp">Processed at: {result['processed_at']}</div>
"""

            html += """
        </div>
    </div>
"""

        # Footer
        html += """
    <div class="footer">
        Report generated by Batch Image Analysis System
    </div>
</body>
</html>
"""

        return html

    def _markdown_to_html(self, text: str) -> str:
        """Convert basic markdown to HTML with table support.

        Args:
            text: Markdown text

        Returns:
            HTML formatted text
        """
        if not text:
            return ""

        import re

        lines = text.split('\n')
        result_lines = []
        in_list = False
        in_table = False
        table_rows = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Check for table (markdown tables have | separators)
            if '|' in stripped and stripped.startswith('|') and stripped.endswith('|'):
                if not in_table:
                    in_table = True
                    table_rows = []

                # Skip separator row (e.g., |---|---|)
                if re.match(r'^\|[\s\-:]+\|', stripped):
                    continue

                # Parse table row
                cells = [cell.strip() for cell in stripped.split('|')[1:-1]]

                # Determine if this is a header row (first row of table)
                is_header = len(table_rows) == 0

                if is_header:
                    row_html = '<tr>' + ''.join([f'<th>{cell}</th>' for cell in cells]) + '</tr>'
                else:
                    row_html = '<tr>' + ''.join([f'<td>{cell}</td>' for cell in cells]) + '</tr>'

                table_rows.append(row_html)
            else:
                # Close table if we were in one
                if in_table:
                    table_html = '<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">'
                    table_html += '<thead>' + table_rows[0] + '</thead>' if table_rows else ''
                    if len(table_rows) > 1:
                        table_html += '<tbody>' + ''.join(table_rows[1:]) + '</tbody>'
                    table_html += '</table>'
                    result_lines.append(table_html)
                    in_table = False
                    table_rows = []

                # Handle headers
                if re.match(r'^##### ', stripped):
                    result_lines.append(re.sub(r'^##### (.+)$', r'<h5>\1</h5>', stripped))
                elif re.match(r'^#### ', stripped):
                    result_lines.append(re.sub(r'^#### (.+)$', r'<h4>\1</h4>', stripped))
                elif re.match(r'^### ', stripped):
                    result_lines.append(re.sub(r'^### (.+)$', r'<h3>\1</h3>', stripped))
                elif re.match(r'^## ', stripped):
                    result_lines.append(re.sub(r'^## (.+)$', r'<h2>\1</h2>', stripped))
                elif re.match(r'^# ', stripped):
                    result_lines.append(re.sub(r'^# (.+)$', r'<h1>\1</h1>', stripped))
                # Handle bullet lists
                elif stripped.startswith('- ') or stripped.startswith('* '):
                    if not in_list:
                        result_lines.append('<ul>')
                        in_list = True
                    result_lines.append(f'<li>{stripped[2:]}</li>')
                else:
                    if in_list:
                        result_lines.append('</ul>')
                        in_list = False
                    result_lines.append(line)

        # Close any open table
        if in_table and table_rows:
            table_html = '<table style="border-collapse: collapse; width: 100%; margin: 10px 0;">'
            table_html += '<thead>' + table_rows[0] + '</thead>' if table_rows else ''
            if len(table_rows) > 1:
                table_html += '<tbody>' + ''.join(table_rows[1:]) + '</tbody>'
            table_html += '</table>'
            result_lines.append(table_html)

        # Close any open list
        if in_list:
            result_lines.append('</ul>')

        text = '\n'.join(result_lines)

        # Convert bold and italic
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)

        # Convert line breaks to paragraphs
        text = re.sub(r'\n\n+', '</p><p>', text)
        text = f'<p>{text}</p>'
        text = text.replace('<p></p>', '')
        # Don't wrap tables and headers in paragraphs
        text = re.sub(r'<p>(<table[^>]*>.*?</table>)</p>', r'\1', text, flags=re.DOTALL)
        text = re.sub(r'<p>(<h[1-6]>.*?</h[1-6]>)</p>', r'\1', text)
        text = re.sub(r'<p>(<ul>.*?</ul>)</p>', r'\1', text, flags=re.DOTALL)

        return text

    async def get_report_metadata(self, batch_id: str) -> Dict[str, Any]:
        """Get metadata about a report without generating it.

        Args:
            batch_id: Batch identifier

        Returns:
            Report metadata
        """
        batch_job = await self.storage.get_batch_job(batch_id)
        if not batch_job:
            return None

        results = await self.storage.get_batch_results(batch_id)

        return {
            "batch_id": batch_id,
            "status": batch_job["status"],
            "total_images": batch_job["total_images"],
            "successful": sum(1 for r in results if not r.get("error_message")),
            "failed": sum(1 for r in results if r.get("error_message")),
            "analysis_prompt": batch_job["analysis_prompt"],
            "created_at": batch_job["created_at"],
            "completed_at": batch_job["completed_at"]
        }
