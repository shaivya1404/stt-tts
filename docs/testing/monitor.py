#!/usr/bin/env python3
"""
TTS/STT Services Monitor
========================

Continuous monitoring script for TTS and STT ML services.

Features:
- Health check monitoring
- Response time tracking
- Automatic alerting
- Status dashboard
- Log file generation

Usage:
    python monitor.py --stt-url https://xxx.ngrok-free.app --tts-url https://yyy.ngrok-free.app
    python monitor.py --stt-url URL --tts-url URL --interval 60 --log monitor.log

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
from typing import Dict, List, Optional, Tuple

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

DEFAULT_INTERVAL = 30  # seconds
DEFAULT_TIMEOUT = 30   # seconds


# =============================================================================
# Color Utilities
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


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


# =============================================================================
# Monitoring Functions
# =============================================================================

class ServiceMonitor:
    """Monitor class for TTS and STT services."""

    def __init__(self, stt_url: str, tts_url: str, log_file: Optional[str] = None):
        self.stt_url = stt_url.rstrip("/")
        self.tts_url = tts_url.rstrip("/")
        self.log_file = log_file

        # History tracking
        self.history: List[Dict] = []
        self.max_history = 100

        # Statistics
        self.stats = {
            "stt": {"checks": 0, "successes": 0, "failures": 0, "total_time": 0},
            "tts": {"checks": 0, "successes": 0, "failures": 0, "total_time": 0}
        }

    def log(self, message: str, level: str = "INFO"):
        """Log a message to file if configured."""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] [{level}] {message}"

        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(log_entry + "\n")
            except Exception as e:
                print(f"Warning: Could not write to log file: {e}")

    def check_health(self, service: str, url: str) -> Tuple[bool, float, Optional[Dict]]:
        """Check health of a service.

        Returns:
            Tuple of (is_healthy, response_time, response_data)
        """
        try:
            start = time.time()
            response = requests.get(
                f"{url}/ml/{service}/health",
                headers=DEFAULT_HEADERS,
                timeout=DEFAULT_TIMEOUT
            )
            elapsed = time.time() - start

            if response.status_code == 200:
                return True, elapsed, response.json()
            else:
                return False, elapsed, {"error": f"HTTP {response.status_code}"}

        except requests.exceptions.Timeout:
            return False, DEFAULT_TIMEOUT, {"error": "Timeout"}
        except requests.exceptions.ConnectionError:
            return False, 0, {"error": "Connection refused"}
        except Exception as e:
            return False, 0, {"error": str(e)}

    def run_check(self) -> Dict:
        """Run a single health check for both services."""
        timestamp = datetime.now()

        # Check STT
        stt_healthy, stt_time, stt_data = self.check_health("stt", self.stt_url)
        self.stats["stt"]["checks"] += 1
        self.stats["stt"]["total_time"] += stt_time
        if stt_healthy:
            self.stats["stt"]["successes"] += 1
        else:
            self.stats["stt"]["failures"] += 1

        # Check TTS
        tts_healthy, tts_time, tts_data = self.check_health("tts", self.tts_url)
        self.stats["tts"]["checks"] += 1
        self.stats["tts"]["total_time"] += tts_time
        if tts_healthy:
            self.stats["tts"]["successes"] += 1
        else:
            self.stats["tts"]["failures"] += 1

        # Create result
        result = {
            "timestamp": timestamp.isoformat(),
            "stt": {
                "healthy": stt_healthy,
                "response_time": round(stt_time, 3),
                "data": stt_data
            },
            "tts": {
                "healthy": tts_healthy,
                "response_time": round(tts_time, 3),
                "data": tts_data
            }
        }

        # Add to history
        self.history.append(result)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # Log
        stt_status = "UP" if stt_healthy else "DOWN"
        tts_status = "UP" if tts_healthy else "DOWN"
        self.log(f"STT: {stt_status} ({stt_time:.2f}s), TTS: {tts_status} ({tts_time:.2f}s)")

        return result

    def get_uptime(self, service: str) -> float:
        """Calculate uptime percentage for a service."""
        stats = self.stats[service]
        if stats["checks"] == 0:
            return 0.0
        return (stats["successes"] / stats["checks"]) * 100

    def get_avg_response_time(self, service: str) -> float:
        """Calculate average response time for a service."""
        stats = self.stats[service]
        if stats["successes"] == 0:
            return 0.0
        return stats["total_time"] / stats["successes"]

    def print_dashboard(self, result: Dict):
        """Print a status dashboard."""
        clear_screen()

        timestamp = result["timestamp"]
        stt = result["stt"]
        tts = result["tts"]

        # Header
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}    TTS/STT SERVICES MONITOR{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")

        print(f"\n{Colors.CYAN}Last Check:{Colors.ENDC} {timestamp}")
        print(f"{Colors.CYAN}Monitoring:{Colors.ENDC}")
        print(f"  STT: {self.stt_url}")
        print(f"  TTS: {self.tts_url}")

        # Current Status
        print(f"\n{Colors.BOLD}Current Status{Colors.ENDC}")
        print("-" * 40)

        # STT Status
        if stt["healthy"]:
            stt_icon = f"{Colors.GREEN}●{Colors.ENDC}"
            stt_status = f"{Colors.GREEN}HEALTHY{Colors.ENDC}"
        else:
            stt_icon = f"{Colors.FAIL}●{Colors.ENDC}"
            stt_status = f"{Colors.FAIL}DOWN{Colors.ENDC}"
            error = stt["data"].get("error", "Unknown")
            stt_status += f" ({error})"

        print(f"  {stt_icon} STT Service: {stt_status}")
        print(f"      Response Time: {stt['response_time']:.2f}s")

        # TTS Status
        if tts["healthy"]:
            tts_icon = f"{Colors.GREEN}●{Colors.ENDC}"
            tts_status = f"{Colors.GREEN}HEALTHY{Colors.ENDC}"
        else:
            tts_icon = f"{Colors.FAIL}●{Colors.ENDC}"
            tts_status = f"{Colors.FAIL}DOWN{Colors.ENDC}"
            error = tts["data"].get("error", "Unknown")
            tts_status += f" ({error})"

        print(f"  {tts_icon} TTS Service: {tts_status}")
        print(f"      Response Time: {tts['response_time']:.2f}s")

        # Statistics
        print(f"\n{Colors.BOLD}Statistics{Colors.ENDC}")
        print("-" * 40)

        stt_uptime = self.get_uptime("stt")
        tts_uptime = self.get_uptime("tts")
        stt_avg = self.get_avg_response_time("stt")
        tts_avg = self.get_avg_response_time("tts")

        uptime_color_stt = Colors.GREEN if stt_uptime >= 95 else (Colors.WARNING if stt_uptime >= 80 else Colors.FAIL)
        uptime_color_tts = Colors.GREEN if tts_uptime >= 95 else (Colors.WARNING if tts_uptime >= 80 else Colors.FAIL)

        print(f"  STT: {uptime_color_stt}{stt_uptime:.1f}% uptime{Colors.ENDC}, "
              f"{self.stats['stt']['checks']} checks, "
              f"avg {stt_avg:.2f}s")
        print(f"  TTS: {uptime_color_tts}{tts_uptime:.1f}% uptime{Colors.ENDC}, "
              f"{self.stats['tts']['checks']} checks, "
              f"avg {tts_avg:.2f}s")

        # Recent History
        print(f"\n{Colors.BOLD}Recent History (last 5 checks){Colors.ENDC}")
        print("-" * 40)

        for entry in self.history[-5:]:
            ts = entry["timestamp"].split("T")[1].split(".")[0]
            stt_ok = "✓" if entry["stt"]["healthy"] else "✗"
            tts_ok = "✓" if entry["tts"]["healthy"] else "✗"
            stt_color = Colors.GREEN if entry["stt"]["healthy"] else Colors.FAIL
            tts_color = Colors.GREEN if entry["tts"]["healthy"] else Colors.FAIL
            print(f"  {ts}  STT: {stt_color}{stt_ok}{Colors.ENDC}  "
                  f"TTS: {tts_color}{tts_ok}{Colors.ENDC}")

        # Footer
        print(f"\n{Colors.CYAN}Press Ctrl+C to stop monitoring{Colors.ENDC}")

    def run(self, interval: int = DEFAULT_INTERVAL, continuous: bool = True):
        """Run the monitor.

        Args:
            interval: Seconds between checks
            continuous: If True, run continuously; otherwise run once
        """
        print(f"Starting monitor (interval={interval}s)...")
        self.log("Monitor started")

        try:
            while True:
                result = self.run_check()
                self.print_dashboard(result)

                if not continuous:
                    break

                time.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n{Colors.WARNING}Monitor stopped by user{Colors.ENDC}")
            self.log("Monitor stopped by user")

        # Final summary
        self.print_summary()

    def print_summary(self):
        """Print final summary."""
        print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.HEADER}{Colors.BOLD}    MONITORING SUMMARY{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")

        print(f"\n{Colors.BOLD}STT Service:{Colors.ENDC}")
        print(f"  Total Checks: {self.stats['stt']['checks']}")
        print(f"  Successes: {self.stats['stt']['successes']}")
        print(f"  Failures: {self.stats['stt']['failures']}")
        print(f"  Uptime: {self.get_uptime('stt'):.1f}%")
        print(f"  Avg Response Time: {self.get_avg_response_time('stt'):.2f}s")

        print(f"\n{Colors.BOLD}TTS Service:{Colors.ENDC}")
        print(f"  Total Checks: {self.stats['tts']['checks']}")
        print(f"  Successes: {self.stats['tts']['successes']}")
        print(f"  Failures: {self.stats['tts']['failures']}")
        print(f"  Uptime: {self.get_uptime('tts'):.1f}%")
        print(f"  Avg Response Time: {self.get_avg_response_time('tts'):.2f}s")

        if self.log_file:
            print(f"\n{Colors.CYAN}Log file: {self.log_file}{Colors.ENDC}")

    def export_history(self, filename: str = "monitor_history.json"):
        """Export monitoring history to JSON file."""
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "stt_url": self.stt_url,
            "tts_url": self.tts_url,
            "statistics": self.stats,
            "history": self.history
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)

        print(f"History exported to: {filename}")


# =============================================================================
# Quick Check Functions
# =============================================================================

def quick_check(stt_url: str, tts_url: str):
    """Run a quick health check and exit."""
    print(f"{Colors.BOLD}Quick Health Check{Colors.ENDC}")
    print("=" * 40)

    # Check STT
    print(f"\n{Colors.CYAN}STT Service:{Colors.ENDC} {stt_url}")
    try:
        response = requests.get(
            f"{stt_url}/ml/stt/health",
            headers=DEFAULT_HEADERS,
            timeout=DEFAULT_TIMEOUT
        )
        if response.status_code == 200:
            data = response.json()
            print(f"  {Colors.GREEN}● HEALTHY{Colors.ENDC}")
            print(f"  Status: {data.get('status')}")
            print(f"  Models: {len(data.get('models', []))} loaded")
        else:
            print(f"  {Colors.FAIL}● HTTP {response.status_code}{Colors.ENDC}")
    except requests.exceptions.ConnectionError:
        print(f"  {Colors.FAIL}● CONNECTION REFUSED{Colors.ENDC}")
    except Exception as e:
        print(f"  {Colors.FAIL}● ERROR: {e}{Colors.ENDC}")

    # Check TTS
    print(f"\n{Colors.CYAN}TTS Service:{Colors.ENDC} {tts_url}")
    try:
        response = requests.get(
            f"{tts_url}/ml/tts/health",
            headers=DEFAULT_HEADERS,
            timeout=DEFAULT_TIMEOUT
        )
        if response.status_code == 200:
            data = response.json()
            print(f"  {Colors.GREEN}● HEALTHY{Colors.ENDC}")
            print(f"  Status: {data.get('status')}")
            print(f"  Models: {len(data.get('models', []))} loaded")
        else:
            print(f"  {Colors.FAIL}● HTTP {response.status_code}{Colors.ENDC}")
    except requests.exceptions.ConnectionError:
        print(f"  {Colors.FAIL}● CONNECTION REFUSED{Colors.ENDC}")
    except Exception as e:
        print(f"  {Colors.FAIL}● ERROR: {e}{Colors.ENDC}")

    print()


# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TTS/STT Services Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick health check
  python monitor.py --stt-url https://xxx.ngrok-free.app --tts-url https://yyy.ngrok-free.app --quick

  # Continuous monitoring (every 30 seconds)
  python monitor.py --stt-url https://xxx.ngrok-free.app --tts-url https://yyy.ngrok-free.app

  # Custom interval with logging
  python monitor.py --stt-url URL --tts-url URL --interval 60 --log monitor.log
        """
    )

    parser.add_argument("--stt-url", required=True, help="STT service URL")
    parser.add_argument("--tts-url", required=True, help="TTS service URL")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                        help=f"Check interval in seconds (default: {DEFAULT_INTERVAL})")
    parser.add_argument("--log", help="Log file path")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Quick check and exit")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable colored output")

    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        Colors.disable()

    if args.quick:
        quick_check(args.stt_url, args.tts_url)
    else:
        monitor = ServiceMonitor(args.stt_url, args.tts_url, args.log)
        monitor.run(interval=args.interval)


if __name__ == "__main__":
    main()
