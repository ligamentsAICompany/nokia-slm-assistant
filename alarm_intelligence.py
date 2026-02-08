"""
Nokia SLM Alarm Intelligence
=============================
Specialized handling for alarm and log queries.
Extracts structured information and provides documented resolutions.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class AlarmSeverity(Enum):
    """Alarm severity levels."""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    WARNING = "warning"
    INFO = "info"
    UNKNOWN = "unknown"


@dataclass
class AlarmInfo:
    """Structured alarm information."""
    alarm_name: str = ""
    alarm_code: str = ""
    severity: AlarmSeverity = AlarmSeverity.UNKNOWN
    source_type: str = ""  # ONT, OLT, UPS, Network
    source_id: str = ""    # e.g., PON[1/1/1], ONT-123
    timestamp: str = ""
    raw_text: str = ""
    
    # Extracted details
    probable_causes: List[str] = field(default_factory=list)
    affected_components: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "alarm_name": self.alarm_name,
            "alarm_code": self.alarm_code,
            "severity": self.severity.value,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "timestamp": self.timestamp,
            "probable_causes": self.probable_causes,
            "affected_components": self.affected_components,
        }


class AlarmParser:
    """
    Parser for extracting alarm information from logs and queries.
    """
    
    # === Alarm Name Patterns ===
    ALARM_PATTERNS = {
        "LOS": r"\bL[Oo]S\b|loss\s+of\s+signal",
        "LOF": r"\bL[Oo]F\b|loss\s+of\s+frame",
        "SF": r"\bSF\b|signal\s+fail",
        "SD": r"\bSD\b|signal\s+degrade",
        "DYING_GASP": r"dying[\s_]?gasp",
        "RANGING_FAILED": r"ranging\s+fail",
        "AUTHENTICATION_FAILED": r"auth(entication)?\s+fail",
        "POWER_FAILURE": r"power\s+(fail|loss|down)",
        "OPTICAL_POWER_LOW": r"optical\s+power\s+(low|high)|[-]?\d+\.?\d*\s*dBm",
        "LINK_DOWN": r"link\s+down|port\s+down",
        "TIMEOUT": r"timeout|timed?\s+out",
        "DEACTIVATION": r"deactivat",
        "TEMPERATURE": r"temperature\s+(high|low|alarm)",
    }
    
    # === Severity Keywords ===
    SEVERITY_KEYWORDS = {
        AlarmSeverity.CRITICAL: ["critical", "fatal", "emergency"],
        AlarmSeverity.MAJOR: ["major", "error", "err"],
        AlarmSeverity.MINOR: ["minor"],
        AlarmSeverity.WARNING: ["warning", "warn"],
        AlarmSeverity.INFO: ["info", "notice", "debug"],
    }
    
    # === Source Type Patterns ===
    SOURCE_PATTERNS = {
        "ONT": r"\bONT[-_]?\d+|\bONU\b",
        "OLT": r"\bOLT[-_]?\d+|\bISAM\b",
        "PON": r"PON\s*\[\d+/\d+/\d+\]",
        "UPS": r"\bUPS\b",
        "NETWORK": r"\bnetwork\b|\bvlan\b|\bbridge\b",
    }
    
    # === Timestamp Patterns ===
    TIMESTAMP_PATTERNS = [
        r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}",  # ISO format
        r"[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}",  # Syslog format
        r"\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}",  # US format
    ]
    
    # === Error Code Patterns ===
    ERROR_CODE_PATTERN = r"(?:error|errno|code)[=:\s]*(0?x?[0-9a-fA-F]+)"
    
    def parse(self, text: str) -> AlarmInfo:
        """
        Parse alarm information from text.
        
        Args:
            text: Raw log or query text.
            
        Returns:
            AlarmInfo with extracted details.
        """
        alarm = AlarmInfo(raw_text=text)
        
        # Extract alarm name
        alarm.alarm_name = self._extract_alarm_name(text)
        
        # Extract severity
        alarm.severity = self._extract_severity(text)
        
        # Extract source
        alarm.source_type, alarm.source_id = self._extract_source(text)
        
        # Extract timestamp
        alarm.timestamp = self._extract_timestamp(text)
        
        # Extract error code
        alarm.alarm_code = self._extract_error_code(text)
        
        # Extract affected components
        alarm.affected_components = self._extract_components(text)
        
        return alarm
    
    def _extract_alarm_name(self, text: str) -> str:
        """Extract the primary alarm name."""
        for name, pattern in self.ALARM_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                return name
        return "UNKNOWN_ALARM"
    
    def _extract_severity(self, text: str) -> AlarmSeverity:
        """Extract alarm severity."""
        text_lower = text.lower()
        for severity, keywords in self.SEVERITY_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    return severity
        return AlarmSeverity.UNKNOWN
    
    def _extract_source(self, text: str) -> Tuple[str, str]:
        """Extract source type and ID."""
        for source_type, pattern in self.SOURCE_PATTERNS.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return source_type, match.group(0)
        return "", ""
    
    def _extract_timestamp(self, text: str) -> str:
        """Extract timestamp from text."""
        for pattern in self.TIMESTAMP_PATTERNS:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return ""
    
    def _extract_error_code(self, text: str) -> str:
        """Extract error code."""
        match = re.search(self.ERROR_CODE_PATTERN, text, re.IGNORECASE)
        if match:
            return match.group(1)
        return ""
    
    def _extract_components(self, text: str) -> List[str]:
        """Extract affected Nokia components."""
        components = []
        component_patterns = [
            r"T-CONT", r"GEM", r"DBA", r"QoS", r"VLAN",
            r"TR-069", r"OMCI", r"PLOAM", r"fiber", r"splitter"
        ]
        for pattern in component_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                components.append(pattern.replace("\\", ""))
        return components


class AlarmResponseFormatter:
    """
    Formatter for structured alarm responses.
    """
    
    RESPONSE_TEMPLATE = """## Alarm Analysis

### Alarm Information
- **Name**: {alarm_name}
- **Code**: {alarm_code}
- **Severity**: {severity}
- **Source**: {source_type} {source_id}
- **Timestamp**: {timestamp}

### Meaning
{meaning}

### Probable Causes
{causes}

### Documented Checks
{checks}

### Documented Resolution
{resolution}
"""
    
    def format_response(
        self, 
        alarm: AlarmInfo, 
        doc_context: str
    ) -> str:
        """
        Format a structured response for an alarm query.
        
        Args:
            alarm: Parsed alarm information.
            doc_context: Retrieved documentation context.
            
        Returns:
            Formatted response string.
        """
        # Extract sections from documentation context
        meaning = self._extract_section(doc_context, "meaning", "definition", "description")
        causes = self._extract_section(doc_context, "cause", "reason", "why")
        checks = self._extract_section(doc_context, "check", "verify", "diagnos")
        resolution = self._extract_section(doc_context, "resolution", "fix", "solution", "action")
        
        return self.RESPONSE_TEMPLATE.format(
            alarm_name=alarm.alarm_name or "Unknown",
            alarm_code=alarm.alarm_code or "N/A",
            severity=alarm.severity.value.upper(),
            source_type=alarm.source_type or "Unknown",
            source_id=alarm.source_id or "",
            timestamp=alarm.timestamp or "Not specified",
            meaning=meaning or "Refer to documentation context.",
            causes=causes or "See documentation for probable causes.",
            checks=checks or "See documentation for diagnostic steps.",
            resolution=resolution or "See documentation for resolution steps.",
        )
    
    def _extract_section(self, context: str, *keywords) -> str:
        """
        Extract a section from context based on keywords.
        
        This is a simple heuristic approach. For production,
        consider using the LLM for extraction.
        """
        lines = context.split('\n')
        section_lines = []
        in_section = False
        
        for line in lines:
            line_lower = line.lower()
            
            # Check if this line starts a relevant section
            if any(kw in line_lower for kw in keywords):
                in_section = True
                section_lines.append(line)
                continue
            
            # Check if we've hit a new section header
            if in_section and line.strip().endswith(':'):
                break
            
            if in_section:
                section_lines.append(line)
        
        if section_lines:
            return '\n'.join(section_lines[:10])  # Limit to 10 lines
        return ""


class AlarmIntelligence:
    """
    Main alarm intelligence handler.
    """
    
    def __init__(self):
        self.parser = AlarmParser()
        self.formatter = AlarmResponseFormatter()
    
    def is_alarm_query(self, query: str) -> bool:
        """Check if a query is alarm-related."""
        alarm_indicators = [
            r"\balarm\b",
            r"\berror\b",
            r"\bfail",
            r"\bLOS\b",
            r"\bLOF\b",
            r"dBm",
            r"ranging",
            r"\d{4}-\d{2}-\d{2}",  # Timestamp
        ]
        for pattern in alarm_indicators:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def extract_search_terms(self, query: str) -> List[str]:
        """
        Extract search terms from an alarm query for retrieval.
        
        Args:
            query: Raw alarm/log query.
            
        Returns:
            List of search terms.
        """
        alarm = self.parser.parse(query)
        terms = []
        
        # Add alarm name
        if alarm.alarm_name and alarm.alarm_name != "UNKNOWN_ALARM":
            terms.append(alarm.alarm_name)
        
        # Add components
        terms.extend(alarm.affected_components)
        
        # Add source type
        if alarm.source_type:
            terms.append(alarm.source_type)
        
        # Add error code
        if alarm.alarm_code:
            terms.append(f"error {alarm.alarm_code}")
        
        # Add generic alarm-related terms
        terms.extend(["troubleshoot", "resolution", "fix"])
        
        # Deduplicate while preserving order
        seen = set()
        unique_terms = []
        for term in terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)
        
        return unique_terms[:15]  # Cap at 15 terms
    
    def process_alarm_query(
        self, 
        query: str, 
        doc_context: str
    ) -> Tuple[AlarmInfo, str]:
        """
        Process an alarm query and generate a structured response.
        
        Args:
            query: Raw alarm/log query.
            doc_context: Retrieved documentation context.
            
        Returns:
            Tuple of (AlarmInfo, formatted_response).
        """
        alarm = self.parser.parse(query)
        response = self.formatter.format_response(alarm, doc_context)
        
        logger.info(
            f"Processed alarm query: {alarm.alarm_name} | "
            f"Severity: {alarm.severity.value} | "
            f"Source: {alarm.source_type}"
        )
        
        return alarm, response


# Global instance
_alarm_intelligence: Optional[AlarmIntelligence] = None


def get_alarm_intelligence() -> AlarmIntelligence:
    """Get or create the global alarm intelligence instance."""
    global _alarm_intelligence
    if _alarm_intelligence is None:
        _alarm_intelligence = AlarmIntelligence()
    return _alarm_intelligence
