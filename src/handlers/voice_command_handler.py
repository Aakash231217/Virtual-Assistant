#!/usr/bin/env python3
"""
Voice Command Handler

Processes and validates voice commands to determine if a shutdown command was spoken.
Supports multiple languages and provides confidence scoring for command recognition.

Author: System
Date: 2024
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class CommandType(Enum):
    """Enumeration of supported command types."""
    SHUTDOWN = "shutdown"
    RESTART = "restart"
    SLEEP = "sleep"
    CANCEL = "cancel"
    UNKNOWN = "unknown"


@dataclass
class VoiceCommandResult:
    """Result of voice command processing."""
    command_type: CommandType
    confidence: float
    raw_text: str
    normalized_text: str
    is_valid: bool
    error_message: Optional[str] = None


class VoiceCommandHandler:
    """Handles processing and validation of voice commands for system control."""
    
    # Shutdown command patterns in multiple languages
    SHUTDOWN_PATTERNS = {
        'english': [
            r'\b(shut\s*down|shutdown|turn\s*off|power\s*off)\b',
            r'\b(shut\s*it\s*down|power\s*down)\b',
            r'\b(switch\s*off|turn\s*off\s*computer)\b'
        ],
        'spanish': [
            r'\b(apagar|cerrar|apaga)\b',
            r'\b(apagar\s*computadora|cerrar\s*sistema)\b'
        ],
        'french': [
            r'\b(éteindre|arrêter|fermer)\b',
            r'\b(éteindre\s*ordinateur|arrêter\s*système)\b'
        ]
    }
    
    RESTART_PATTERNS = {
        'english': [
            r'\b(restart|reboot|reset)\b',
            r'\b(restart\s*computer|reboot\s*system)\b'
        ]
    }
    
    SLEEP_PATTERNS = {
        'english': [
            r'\b(sleep|hibernate|suspend)\b',
            r'\b(go\s*to\s*sleep|put\s*to\s*sleep)\b'
        ]
    }
    
    CANCEL_PATTERNS = {
        'english': [
            r'\b(cancel|stop|abort|nevermind|never\s*mind)\b',
            r'\b(don\'t\s*do\s*it|cancel\s*that)\b'
        ]
    }
    
    # Confidence thresholds
    MIN_CONFIDENCE_THRESHOLD = 0.6
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    
    def __init__(self, min_confidence: float = 0.6, enable_logging: bool = True):
        """
        Initialize the voice command handler.
        
        Args:
            min_confidence: Minimum confidence threshold for command acceptance
            enable_logging: Whether to enable logging
        """
        self.min_confidence = min_confidence
        self.logger = self._setup_logger() if enable_logging else None
        self._command_history: List[VoiceCommandResult] = []
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the handler."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def process_voice_command(self, raw_text: str, language: str = 'english') -> VoiceCommandResult:
        """
        Process a voice command and determine its type and validity.
        
        Args:
            raw_text: The raw text from voice recognition
            language: Language of the command (default: 'english')
        
        Returns:
            VoiceCommandResult containing processing results
        """
        try:
            if self.logger:
                self.logger.info(f"Processing voice command: '{raw_text}'")
            
            # Validate input
            if not raw_text or not isinstance(raw_text, str):
                return VoiceCommandResult(
                    command_type=CommandType.UNKNOWN,
                    confidence=0.0,
                    raw_text=raw_text or "",
                    normalized_text="",
                    is_valid=False,
                    error_message="Invalid or empty input text"
                )
            
            # Normalize the text
            normalized_text = self._normalize_text(raw_text)
            
            # Detect command type and confidence
            command_type, confidence = self._detect_command_type(normalized_text, language)
            
            # Determine if command is valid
            is_valid = confidence >= self.min_confidence
            
            result = VoiceCommandResult(
                command_type=command_type,
                confidence=confidence,
                raw_text=raw_text,
                normalized_text=normalized_text,
                is_valid=is_valid,
                error_message=None if is_valid else f"Confidence too low: {confidence:.2f}"
            )
            
            # Store in history
            self._command_history.append(result)
            
            if self.logger:
                self.logger.info(
                    f"Command processed: {command_type.value}, "
                    f"confidence: {confidence:.2f}, valid: {is_valid}"
                )
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing voice command: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            
            return VoiceCommandResult(
                command_type=CommandType.UNKNOWN,
                confidence=0.0,
                raw_text=raw_text,
                normalized_text="",
                is_valid=False,
                error_message=error_msg
            )
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for better pattern matching.
        
        Args:
            text: Raw input text
        
        Returns:
            Normalized text
        """
        # Convert to lowercase
        normalized = text.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove punctuation except apostrophes
        normalized = re.sub(r'[^\w\s\']', ' ', normalized)
        
        return normalized
    
    def _detect_command_type(self, text: str, language: str) -> Tuple[CommandType, float]:
        """
        Detect the command type and calculate confidence score.
        
        Args:
            text: Normalized text to analyze
            language: Language for pattern matching
        
        Returns:
            Tuple of (CommandType, confidence_score)
        """
        max_confidence = 0.0
        detected_command = CommandType.UNKNOWN
        
        # Check each command type
        command_checks = [
            (CommandType.SHUTDOWN, self.SHUTDOWN_PATTERNS),
            (CommandType.RESTART, self.RESTART_PATTERNS),
            (CommandType.SLEEP, self.SLEEP_PATTERNS),
            (CommandType.CANCEL, self.CANCEL_PATTERNS)
        ]
        
        for command_type, patterns_dict in command_checks:
            if language in patterns_dict:
                confidence = self._calculate_pattern_confidence(
                    text, patterns_dict[language]
                )
                if confidence > max_confidence:
                    max_confidence = confidence
                    detected_command = command_type
        
        return detected_command, max_confidence
    
    def _calculate_pattern_confidence(self, text: str, patterns: List[str]) -> float:
        """
        Calculate confidence score for pattern matching.
        
        Args:
            text: Text to match against
            patterns: List of regex patterns
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        max_confidence = 0.0
        
        for pattern in patterns:
            try:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Calculate confidence based on match quality
                    match_length = sum(len(match) if isinstance(match, str) else len(' '.join(match)) for match in matches)
                    text_length = len(text)
                    
                    # Base confidence from match ratio
                    confidence = min(match_length / max(text_length, 1), 1.0)
                    
                    # Boost confidence for exact matches
                    if any(match.strip() == text.strip() for match in matches if isinstance(match, str)):
                        confidence = min(confidence * 1.5, 1.0)
                    
                    # Boost confidence for multiple matches
                    if len(matches) > 1:
                        confidence = min(confidence * 1.2, 1.0)
                    
                    max_confidence = max(max_confidence, confidence)
                    
            except re.error as e:
                if self.logger:
                    self.logger.warning(f"Invalid regex pattern: {pattern}, error: {e}")
                continue
        
        return max_confidence
    
    def is_shutdown_command(self, text: str, language: str = 'english') -> bool:
        """
        Quick check if the given text is a shutdown command.
        
        Args:
            text: Text to check
            language: Language for pattern matching
        
        Returns:
            True if it's a valid shutdown command
        """
        result = self.process_voice_command(text, language)
        return result.is_valid and result.command_type == CommandType.SHUTDOWN
    
    def get_command_history(self, limit: Optional[int] = None) -> List[VoiceCommandResult]:
        """
        Get the history of processed commands.
        
        Args:
            limit: Maximum number of commands to return
        
        Returns:
            List of VoiceCommandResult objects
        """
        if limit is None:
            return self._command_history.copy()
        return self._command_history[-limit:]
    
    def clear_history(self) -> None:
        """Clear the command history."""
        self._command_history.clear()
        if self.logger:
            self.logger.info("Command history cleared")
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.
        
        Returns:
            List of supported language codes
        """
        return list(self.SHUTDOWN_PATTERNS.keys())
    
    def add_custom_pattern(self, command_type: CommandType, language: str, pattern: str) -> bool:
        """
        Add a custom pattern for command recognition.
        
        Args:
            command_type: Type of command
            language: Language code
            pattern: Regex pattern to add
        
        Returns:
            True if pattern was added successfully
        """
        try:
            # Validate pattern
            re.compile(pattern)
            
            # Get the appropriate pattern dictionary
            pattern_dict = None
            if command_type == CommandType.SHUTDOWN:
                pattern_dict = self.SHUTDOWN_PATTERNS
            elif command_type == CommandType.RESTART:
                pattern_dict = self.RESTART_PATTERNS
            elif command_type == CommandType.SLEEP:
                pattern_dict = self.SLEEP_PATTERNS
            elif command_type == CommandType.CANCEL:
                pattern_dict = self.CANCEL_PATTERNS
            
            if pattern_dict is not None:
                if language not in pattern_dict:
                    pattern_dict[language] = []
                pattern_dict[language].append(pattern)
                
                if self.logger:
                    self.logger.info(f"Added custom pattern for {command_type.value} in {language}")
                return True
            
        except re.error as e:
            if self.logger:
                self.logger.error(f"Invalid regex pattern: {pattern}, error: {e}")
        
        return False


# Example usage and testing
if __name__ == "__main__":
    # Create handler instance
    handler = VoiceCommandHandler(min_confidence=0.6)
    
    # Test commands
    test_commands = [
        "shutdown the computer",
        "turn off the system",
        "restart now",
        "go to sleep",
        "cancel that",
        "play some music",  # Should not match
        "shut down",
        "power off"
    ]
    
    print("Testing Voice Command Handler:")
    print("=" * 40)
    
    for command in test_commands:
        result = handler.process_voice_command(command)
        print(f"Command: '{command}'")
        print(f"  Type: {result.command_type.value}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Valid: {result.is_valid}")
        if result.error_message:
            print(f"  Error: {result.error_message}")
        print()
    
    # Test shutdown detection
    print("Shutdown Command Tests:")
    print("-" * 25)
    shutdown_tests = ["shutdown", "turn off computer", "hello world"]
    for test in shutdown_tests:
        is_shutdown = handler.is_shutdown_command(test)
        print(f"'{test}' -> Shutdown: {is_shutdown}")