#!/usr/bin/env python3
"""
TTS/STT Services Test Script
============================

Comprehensive testing script for TTS and STT ML services.

Usage:
    python test_services.py --stt-url https://xxx.ngrok-free.app --tts-url https://yyy.ngrok-free.app
    python test_services.py --interactive

Requirements:
    pip install requests

Author: TTS-STT Team
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    print("ERROR: requests library not installed. Run: pip install requests")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_HEADERS = {
    "ngrok-skip-browser-warning": "true"
}

# Test data
TEST_TEXTS = {
    "en": "Hello, this is a test of the text to speech system.",
    "hi": "नमस्ते, यह टेक्स्ट टू स्पीच सिस्टम का परीक्षण है।",
    "es": "Hola, esta es una prueba del sistema de texto a voz.",
    "fr": "Bonjour, ceci est un test du système de synthèse vocale.",
    "de": "Hallo, dies ist ein Test des Text-zu-Sprache-Systems.",
    "ta": "வணக்கம், இது உரையிலிருந்து பேச்சு அமைப்பின் சோதனை.",
    "te": "హలో, ఇది టెక్స్ట్ టు స్పీచ్ సిస్టమ్ యొక్క పరీక్ష.",
    "ja": "こんにちは、これはテキスト読み上げシステムのテストです。",
    "zh-cn": "你好，这是文字转语音系统的测试。",
}


# =============================================================================
# Utility Functions
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

    @classmethod
    def disable(cls):
        """Disable colors for non-TTY output."""
        cls.HEADER = ''
        cls.BLUE = ''
        cls.CYAN = ''
        cls.GREEN = ''
        cls.WARNING = ''
        cls.FAIL = ''
        cls.ENDC = ''
        cls.BOLD = ''


def print_header(text: str):
    """Print a section header."""
    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.WARNING}! {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.CYAN}→ {text}{Colors.ENDC}")


# =============================================================================
# Test Functions
# =============================================================================

class ServiceTester:
    """Test class for TTS and STT services."""

    def __init__(self, stt_url: str, tts_url: str):
        self.stt_url = stt_url.rstrip("/")
        self.tts_url = tts_url.rstrip("/")
        self.results: List[Dict[str, Any]] = []

    def _record_result(self, test_name: str, success: bool, details: Dict[str, Any]):
        """Record a test result."""
        self.results.append({
            "test": test_name,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            **details
        })

    # -------------------------------------------------------------------------
    # Health Checks
    # -------------------------------------------------------------------------

    def test_stt_health(self) -> bool:
        """Test STT service health endpoint."""
        print_info("Testing STT health endpoint...")

        try:
            start = time.time()
            response = requests.get(
                f"{self.stt_url}/ml/stt/health",
                headers=DEFAULT_HEADERS,
                timeout=30
            )
            elapsed = time.time() - start

            if response.status_code == 200:
                data = response.json()
                print_success(f"STT health OK ({elapsed:.2f}s)")
                print(f"   Status: {data.get('status')}")
                print(f"   Models: {len(data.get('models', []))} loaded")
                self._record_result("stt_health", True, {"response_time": elapsed})
                return True
            else:
                print_error(f"STT health failed: HTTP {response.status_code}")
                self._record_result("stt_health", False, {"status_code": response.status_code})
                return False

        except requests.exceptions.ConnectionError:
            print_error("STT service not reachable - check URL")
            self._record_result("stt_health", False, {"error": "connection_error"})
            return False
        except Exception as e:
            print_error(f"STT health check error: {e}")
            self._record_result("stt_health", False, {"error": str(e)})
            return False

    def test_tts_health(self) -> bool:
        """Test TTS service health endpoint."""
        print_info("Testing TTS health endpoint...")

        try:
            start = time.time()
            response = requests.get(
                f"{self.tts_url}/ml/tts/health",
                headers=DEFAULT_HEADERS,
                timeout=30
            )
            elapsed = time.time() - start

            if response.status_code == 200:
                data = response.json()
                print_success(f"TTS health OK ({elapsed:.2f}s)")
                print(f"   Status: {data.get('status')}")
                print(f"   Models: {len(data.get('models', []))} loaded")
                self._record_result("tts_health", True, {"response_time": elapsed})
                return True
            else:
                print_error(f"TTS health failed: HTTP {response.status_code}")
                self._record_result("tts_health", False, {"status_code": response.status_code})
                return False

        except requests.exceptions.ConnectionError:
            print_error("TTS service not reachable - check URL")
            self._record_result("tts_health", False, {"error": "connection_error"})
            return False
        except Exception as e:
            print_error(f"TTS health check error: {e}")
            self._record_result("tts_health", False, {"error": str(e)})
            return False

    # -------------------------------------------------------------------------
    # STT Tests
    # -------------------------------------------------------------------------

    def test_stt_transcribe(self, audio_file: str, language_hint: str = "en") -> Optional[Dict]:
        """Test STT transcription with an audio file."""
        print_info(f"Testing STT transcription ({language_hint})...")

        if not os.path.exists(audio_file):
            print_warning(f"Audio file not found: {audio_file}")
            print_warning("Skipping transcription test")
            return None

        try:
            start = time.time()
            with open(audio_file, "rb") as f:
                response = requests.post(
                    f"{self.stt_url}/ml/stt/transcribe",
                    files={"file": f},
                    data={"language_hint": language_hint},
                    headers=DEFAULT_HEADERS,
                    timeout=120
                )
            elapsed = time.time() - start

            if response.status_code == 200:
                data = response.json()
                print_success(f"STT transcription OK ({elapsed:.2f}s)")
                print(f"   Text: {data.get('text', '')[:80]}...")
                print(f"   Language: {data.get('language')}")
                print(f"   Confidence: {data.get('confidence')}")
                print(f"   Words: {len(data.get('timestamps', []))}")

                self._record_result("stt_transcribe", True, {
                    "response_time": elapsed,
                    "text_length": len(data.get('text', '')),
                    "confidence": data.get('confidence')
                })
                return data
            else:
                print_error(f"STT transcription failed: HTTP {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                self._record_result("stt_transcribe", False, {"status_code": response.status_code})
                return None

        except Exception as e:
            print_error(f"STT transcription error: {e}")
            self._record_result("stt_transcribe", False, {"error": str(e)})
            return None

    # -------------------------------------------------------------------------
    # TTS Tests
    # -------------------------------------------------------------------------

    def test_tts_synthesize(self, text: str, language: str = "en", speed: float = 1.0) -> Optional[Dict]:
        """Test TTS synthesis."""
        print_info(f"Testing TTS synthesis ({language})...")

        try:
            start = time.time()
            response = requests.post(
                f"{self.tts_url}/ml/tts/predict",
                json={
                    "text": text,
                    "language": language,
                    "speed": speed
                },
                headers={
                    "Content-Type": "application/json",
                    **DEFAULT_HEADERS
                },
                timeout=120
            )
            elapsed = time.time() - start

            if response.status_code == 200:
                data = response.json()
                print_success(f"TTS synthesis OK ({elapsed:.2f}s)")
                print(f"   Duration: {data.get('duration')}s")
                print(f"   Status: {data.get('status')}")
                print(f"   Audio path: {data.get('audio_path', 'N/A')[:50]}...")

                self._record_result("tts_synthesize", True, {
                    "response_time": elapsed,
                    "duration": data.get('duration'),
                    "language": language
                })
                return data
            else:
                print_error(f"TTS synthesis failed: HTTP {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                self._record_result("tts_synthesize", False, {"status_code": response.status_code})
                return None

        except Exception as e:
            print_error(f"TTS synthesis error: {e}")
            self._record_result("tts_synthesize", False, {"error": str(e)})
            return None

    def test_tts_multi_language(self) -> Dict[str, bool]:
        """Test TTS with multiple languages."""
        print_info("Testing TTS multi-language support...")
        results = {}

        for lang, text in TEST_TEXTS.items():
            try:
                response = requests.post(
                    f"{self.tts_url}/ml/tts/predict",
                    json={"text": text, "language": lang},
                    headers={
                        "Content-Type": "application/json",
                        **DEFAULT_HEADERS
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    data = response.json()
                    print_success(f"  {lang}: OK ({data.get('duration')}s)")
                    results[lang] = True
                else:
                    print_error(f"  {lang}: Failed (HTTP {response.status_code})")
                    results[lang] = False

            except Exception as e:
                print_error(f"  {lang}: Error ({e})")
                results[lang] = False

        self._record_result("tts_multi_language", all(results.values()), {"results": results})
        return results

    def test_tts_speed_variations(self) -> Dict[float, bool]:
        """Test TTS with different speeds."""
        print_info("Testing TTS speed variations...")
        results = {}
        text = "Testing speed control."

        for speed in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            try:
                response = requests.post(
                    f"{self.tts_url}/ml/tts/predict",
                    json={"text": text, "language": "en", "speed": speed},
                    headers={
                        "Content-Type": "application/json",
                        **DEFAULT_HEADERS
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    data = response.json()
                    print_success(f"  Speed {speed}x: {data.get('duration')}s")
                    results[speed] = True
                else:
                    print_error(f"  Speed {speed}x: Failed")
                    results[speed] = False

            except Exception as e:
                print_error(f"  Speed {speed}x: Error ({e})")
                results[speed] = False

        self._record_result("tts_speed_variations", all(results.values()), {"results": results})
        return results

    # -------------------------------------------------------------------------
    # Integration Tests
    # -------------------------------------------------------------------------

    def test_round_trip(self, audio_file: str) -> bool:
        """Test STT -> TTS round trip."""
        print_info("Testing round-trip (STT -> TTS)...")

        # Step 1: Transcribe
        stt_result = self.test_stt_transcribe(audio_file)
        if not stt_result:
            print_error("Round-trip failed at STT step")
            return False

        # Step 2: Synthesize
        tts_result = self.test_tts_synthesize(
            stt_result['text'],
            stt_result.get('language', 'en')
        )
        if not tts_result:
            print_error("Round-trip failed at TTS step")
            return False

        print_success("Round-trip test completed successfully!")
        self._record_result("round_trip", True, {
            "stt_text_length": len(stt_result['text']),
            "tts_duration": tts_result.get('duration')
        })
        return True

    # -------------------------------------------------------------------------
    # Performance Tests
    # -------------------------------------------------------------------------

    def test_latency(self, num_runs: int = 3) -> Dict[str, float]:
        """Measure service latency."""
        print_info(f"Measuring latency ({num_runs} runs)...")

        tts_times = []
        text = "Quick latency test."

        for i in range(num_runs):
            try:
                start = time.time()
                response = requests.post(
                    f"{self.tts_url}/ml/tts/predict",
                    json={"text": text, "language": "en"},
                    headers={
                        "Content-Type": "application/json",
                        **DEFAULT_HEADERS
                    },
                    timeout=60
                )
                elapsed = time.time() - start

                if response.status_code == 200:
                    tts_times.append(elapsed)
                    print(f"   Run {i+1}: {elapsed:.2f}s")

            except Exception as e:
                print_error(f"   Run {i+1}: Error ({e})")

        if tts_times:
            stats = {
                "min": min(tts_times),
                "max": max(tts_times),
                "avg": sum(tts_times) / len(tts_times)
            }
            print_success(f"TTS Latency - Min: {stats['min']:.2f}s, Max: {stats['max']:.2f}s, Avg: {stats['avg']:.2f}s")
            self._record_result("latency", True, stats)
            return stats

        return {}

    # -------------------------------------------------------------------------
    # Report Generation
    # -------------------------------------------------------------------------

    def generate_report(self, output_file: str = "test_report.json"):
        """Generate a test report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "stt_url": self.stt_url,
            "tts_url": self.tts_url,
            "total_tests": len(self.results),
            "passed": sum(1 for r in self.results if r['success']),
            "failed": sum(1 for r in self.results if not r['success']),
            "results": self.results
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print_info(f"Report saved to: {output_file}")
        return report

    def print_summary(self):
        """Print test summary."""
        print_header("TEST SUMMARY")

        passed = sum(1 for r in self.results if r['success'])
        failed = sum(1 for r in self.results if not r['success'])
        total = len(self.results)

        print(f"\nTotal Tests: {total}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.ENDC}")
        print(f"{Colors.FAIL}Failed: {failed}{Colors.ENDC}")

        if failed > 0:
            print(f"\n{Colors.FAIL}Failed Tests:{Colors.ENDC}")
            for r in self.results:
                if not r['success']:
                    print(f"  - {r['test']}")


# =============================================================================
# Main Function
# =============================================================================

def run_all_tests(stt_url: str, tts_url: str, audio_file: Optional[str] = None):
    """Run all tests."""
    tester = ServiceTester(stt_url, tts_url)

    print_header("TTS/STT SERVICE TESTS")
    print(f"\nSTT URL: {stt_url}")
    print(f"TTS URL: {tts_url}")
    print(f"Audio File: {audio_file or 'Not provided'}")

    # Health checks
    print_header("HEALTH CHECKS")
    stt_healthy = tester.test_stt_health()
    tts_healthy = tester.test_tts_health()

    if not stt_healthy and not tts_healthy:
        print_error("\nBoth services are unreachable. Aborting tests.")
        return

    # TTS Tests
    if tts_healthy:
        print_header("TTS TESTS")
        tester.test_tts_synthesize("Hello, this is a basic TTS test.", "en")
        tester.test_tts_synthesize("नमस्ते, यह हिंदी परीक्षण है।", "hi")
        tester.test_tts_multi_language()
        tester.test_tts_speed_variations()

    # STT Tests
    if stt_healthy and audio_file:
        print_header("STT TESTS")
        tester.test_stt_transcribe(audio_file, "en")

    # Integration Tests
    if stt_healthy and tts_healthy and audio_file:
        print_header("INTEGRATION TESTS")
        tester.test_round_trip(audio_file)

    # Performance Tests
    if tts_healthy:
        print_header("PERFORMANCE TESTS")
        tester.test_latency(num_runs=3)

    # Summary
    tester.print_summary()
    tester.generate_report()


def interactive_mode():
    """Run in interactive mode."""
    print_header("TTS/STT SERVICE TESTER - Interactive Mode")

    stt_url = input("\nEnter STT URL (e.g., https://xxx.ngrok-free.app): ").strip()
    tts_url = input("Enter TTS URL (e.g., https://yyy.ngrok-free.app): ").strip()
    audio_file = input("Enter audio file path (or press Enter to skip): ").strip() or None

    if not stt_url or not tts_url:
        print_error("Both URLs are required!")
        return

    run_all_tests(stt_url, tts_url, audio_file)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TTS/STT Services Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_services.py --stt-url https://xxx.ngrok-free.app --tts-url https://yyy.ngrok-free.app
  python test_services.py --stt-url https://xxx.ngrok-free.app --tts-url https://yyy.ngrok-free.app --audio sample.wav
  python test_services.py --interactive
        """
    )

    parser.add_argument("--stt-url", help="STT service URL")
    parser.add_argument("--tts-url", help="TTS service URL")
    parser.add_argument("--audio", help="Audio file for STT testing")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")

    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        Colors.disable()

    if args.interactive:
        interactive_mode()
    elif args.stt_url and args.tts_url:
        run_all_tests(args.stt_url, args.tts_url, args.audio)
    else:
        parser.print_help()
        print("\n" + "="*60)
        print("TIP: Use --interactive for guided testing")
        print("="*60)


if __name__ == "__main__":
    main()
