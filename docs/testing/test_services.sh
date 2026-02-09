#!/bin/bash
# =============================================================================
# TTS/STT Services Test Script (Bash)
# =============================================================================
#
# Usage:
#   ./test_services.sh <STT_URL> <TTS_URL>
#   ./test_services.sh https://xxx.ngrok-free.app https://yyy.ngrok-free.app
#
# Requirements:
#   - curl
#   - jq (optional, for pretty JSON output)
#
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check if jq is available
HAS_JQ=$(command -v jq &> /dev/null && echo "true" || echo "false")

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}! $1${NC}"
}

print_info() {
    echo -e "${CYAN}→ $1${NC}"
}

format_json() {
    if [ "$HAS_JQ" = "true" ]; then
        echo "$1" | jq .
    else
        echo "$1"
    fi
}

# =============================================================================
# Test Functions
# =============================================================================

test_stt_health() {
    print_info "Testing STT health endpoint..."

    RESPONSE=$(curl -s -w "\n%{http_code}" \
        -H "ngrok-skip-browser-warning: true" \
        "$STT_URL/ml/stt/health" 2>/dev/null)

    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | sed '$d')

    if [ "$HTTP_CODE" = "200" ]; then
        print_success "STT health OK (HTTP $HTTP_CODE)"
        if [ "$HAS_JQ" = "true" ]; then
            echo "   Status: $(echo "$BODY" | jq -r '.status')"
        fi
        return 0
    else
        print_error "STT health failed (HTTP $HTTP_CODE)"
        return 1
    fi
}

test_tts_health() {
    print_info "Testing TTS health endpoint..."

    RESPONSE=$(curl -s -w "\n%{http_code}" \
        -H "ngrok-skip-browser-warning: true" \
        "$TTS_URL/ml/tts/health" 2>/dev/null)

    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | sed '$d')

    if [ "$HTTP_CODE" = "200" ]; then
        print_success "TTS health OK (HTTP $HTTP_CODE)"
        if [ "$HAS_JQ" = "true" ]; then
            echo "   Status: $(echo "$BODY" | jq -r '.status')"
        fi
        return 0
    else
        print_error "TTS health failed (HTTP $HTTP_CODE)"
        return 1
    fi
}

test_stt_transcribe() {
    local AUDIO_FILE=$1
    local LANG=${2:-en}

    print_info "Testing STT transcription ($LANG)..."

    if [ ! -f "$AUDIO_FILE" ]; then
        print_warning "Audio file not found: $AUDIO_FILE"
        return 1
    fi

    RESPONSE=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "ngrok-skip-browser-warning: true" \
        -F "file=@$AUDIO_FILE" \
        -F "language_hint=$LANG" \
        "$STT_URL/ml/stt/transcribe" 2>/dev/null)

    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | sed '$d')

    if [ "$HTTP_CODE" = "200" ]; then
        print_success "STT transcription OK (HTTP $HTTP_CODE)"
        if [ "$HAS_JQ" = "true" ]; then
            echo "   Text: $(echo "$BODY" | jq -r '.text' | head -c 80)..."
            echo "   Language: $(echo "$BODY" | jq -r '.language')"
            echo "   Confidence: $(echo "$BODY" | jq -r '.confidence')"
        else
            echo "   Response: $BODY"
        fi
        return 0
    else
        print_error "STT transcription failed (HTTP $HTTP_CODE)"
        echo "   Response: $BODY"
        return 1
    fi
}

test_tts_synthesize() {
    local TEXT=$1
    local LANG=${2:-en}
    local OUTPUT_FILE=${3:-/dev/null}

    print_info "Testing TTS synthesis ($LANG)..."

    RESPONSE=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -H "ngrok-skip-browser-warning: true" \
        -d "{\"text\": \"$TEXT\", \"language\": \"$LANG\"}" \
        "$TTS_URL/ml/tts/predict" 2>/dev/null)

    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | sed '$d')

    if [ "$HTTP_CODE" = "200" ]; then
        print_success "TTS synthesis OK (HTTP $HTTP_CODE)"
        if [ "$HAS_JQ" = "true" ]; then
            echo "   Duration: $(echo "$BODY" | jq -r '.duration')s"
            echo "   Status: $(echo "$BODY" | jq -r '.status')"
        else
            echo "   Response: $BODY"
        fi
        return 0
    else
        print_error "TTS synthesis failed (HTTP $HTTP_CODE)"
        echo "   Response: $BODY"
        return 1
    fi
}

test_tts_multi_language() {
    print_info "Testing TTS multi-language support..."

    declare -A TEXTS
    TEXTS["en"]="Hello, this is a test."
    TEXTS["hi"]="नमस्ते, यह एक परीक्षण है।"
    TEXTS["es"]="Hola, esta es una prueba."
    TEXTS["fr"]="Bonjour, ceci est un test."

    local PASSED=0
    local FAILED=0

    for LANG in "${!TEXTS[@]}"; do
        TEXT="${TEXTS[$LANG]}"

        RESPONSE=$(curl -s -w "\n%{http_code}" \
            -X POST \
            -H "Content-Type: application/json" \
            -H "ngrok-skip-browser-warning: true" \
            -d "{\"text\": \"$TEXT\", \"language\": \"$LANG\"}" \
            "$TTS_URL/ml/tts/predict" 2>/dev/null)

        HTTP_CODE=$(echo "$RESPONSE" | tail -n1)

        if [ "$HTTP_CODE" = "200" ]; then
            print_success "  $LANG: OK"
            ((PASSED++))
        else
            print_error "  $LANG: Failed (HTTP $HTTP_CODE)"
            ((FAILED++))
        fi
    done

    echo ""
    echo "   Passed: $PASSED, Failed: $FAILED"
}

# =============================================================================
# Main Script
# =============================================================================

main() {
    # Check arguments
    if [ $# -lt 2 ]; then
        echo "Usage: $0 <STT_URL> <TTS_URL> [AUDIO_FILE]"
        echo ""
        echo "Examples:"
        echo "  $0 https://xxx.ngrok-free.app https://yyy.ngrok-free.app"
        echo "  $0 https://xxx.ngrok-free.app https://yyy.ngrok-free.app sample.wav"
        exit 1
    fi

    STT_URL="${1%/}"  # Remove trailing slash
    TTS_URL="${2%/}"
    AUDIO_FILE="${3:-}"

    print_header "TTS/STT SERVICE TESTS"
    echo ""
    echo "STT URL: $STT_URL"
    echo "TTS URL: $TTS_URL"
    echo "Audio File: ${AUDIO_FILE:-Not provided}"
    echo "jq available: $HAS_JQ"

    # Track results
    PASSED=0
    FAILED=0

    # Health Checks
    print_header "HEALTH CHECKS"

    if test_stt_health; then
        ((PASSED++))
    else
        ((FAILED++))
    fi

    if test_tts_health; then
        ((PASSED++))
    else
        ((FAILED++))
    fi

    # TTS Tests
    print_header "TTS TESTS"

    if test_tts_synthesize "Hello, this is a basic TTS test." "en"; then
        ((PASSED++))
    else
        ((FAILED++))
    fi

    if test_tts_synthesize "नमस्ते, यह हिंदी परीक्षण है।" "hi"; then
        ((PASSED++))
    else
        ((FAILED++))
    fi

    # Multi-language test
    test_tts_multi_language

    # STT Tests
    if [ -n "$AUDIO_FILE" ]; then
        print_header "STT TESTS"

        if test_stt_transcribe "$AUDIO_FILE" "en"; then
            ((PASSED++))
        else
            ((FAILED++))
        fi
    else
        print_header "STT TESTS"
        print_warning "No audio file provided - skipping transcription tests"
        print_info "Provide an audio file as third argument to test STT"
    fi

    # Summary
    print_header "TEST SUMMARY"
    echo ""
    echo "Total Tests: $((PASSED + FAILED))"
    echo -e "${GREEN}Passed: $PASSED${NC}"
    echo -e "${RED}Failed: $FAILED${NC}"

    if [ $FAILED -gt 0 ]; then
        exit 1
    fi
}

# Run main function
main "$@"
