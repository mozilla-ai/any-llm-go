package gemini

import (
	"strings"

	"google.golang.org/genai"

	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

const (
	// Gemini 2.5 exposes token ranges rather than named efforts. These values
	// preserve the adapter's low/medium/high policy while respecting each
	// model family's documented minimum and maximum.
	thinkingBudgetHigh        int32 = 24576
	thinkingBudgetLow         int32 = 1024
	thinkingBudgetMax         int32 = 32768
	thinkingBudgetMedium      int32 = 8192
	thinkingBudgetMinimal     int32 = 256
	thinkingBudgetMinimalLite int32 = 512
	thinkingBudgetProMinimal  int32 = 128
)

// applyThinking uses thinkingLevel for Gemini 3 and thinkingBudget for the
// documented budget-based models. The service validates each Gemini 3 model's
// supported levels so a hard-coded catalog cannot reject new models or go stale.
// https://ai.google.dev/gemini-api/docs/generate-content/thinking
func applyThinking(
	cfg *genai.GenerateContentConfig,
	model string,
	effort providers.ReasoningEffort,
) error {
	if effort == "" || effort == providers.ReasoningEffortAuto {
		return nil
	}

	if usesThinkingLevel(model) {
		level, ok := thinkingLevel(effort)
		if !ok {
			return errors.NewUnsupportedParamError(providerName, "reasoning_effort")
		}
		cfg.ThinkingConfig = &genai.ThinkingConfig{
			IncludeThoughts: true,
			ThinkingLevel:   level,
		}
		return nil
	}

	if !usesThinkingBudget(model) {
		if effort == providers.ReasoningEffortNone {
			return nil
		}
		return errors.NewUnsupportedParamError(providerName, "reasoning_effort")
	}

	budget, ok := thinkingBudget(model, effort)
	if !ok {
		return errors.NewUnsupportedParamError(providerName, "reasoning_effort")
	}
	cfg.ThinkingConfig = &genai.ThinkingConfig{
		IncludeThoughts: effort != providers.ReasoningEffortNone,
		ThinkingBudget:  &budget,
	}
	return nil
}

func thinkingLevel(effort providers.ReasoningEffort) (genai.ThinkingLevel, bool) {
	switch effort {
	case providers.ReasoningEffortAuto, providers.ReasoningEffortNone:
		return "", false
	case providers.ReasoningEffortMinimal:
		return genai.ThinkingLevelMinimal, true
	case providers.ReasoningEffortLow:
		return genai.ThinkingLevelLow, true
	case providers.ReasoningEffortMedium:
		return genai.ThinkingLevelMedium, true
	// Gemini's highest documented level is high, so the normalized maxima
	// collapse to high instead of sending values the API does not accept.
	case providers.ReasoningEffortHigh, providers.ReasoningEffortXHigh, providers.ReasoningEffortMax:
		return genai.ThinkingLevelHigh, true
	default:
		return "", false
	}
}

func thinkingBudget(model string, effort providers.ReasoningEffort) (int32, bool) {
	model = geminiModelName(model)
	switch effort {
	case providers.ReasoningEffortAuto:
		return 0, false
	case providers.ReasoningEffortNone:
		return 0, !strings.Contains(model, "pro")
	case providers.ReasoningEffortMinimal:
		return minimalThinkingBudget(model), true
	case providers.ReasoningEffortLow:
		return thinkingBudgetLow, true
	case providers.ReasoningEffortMedium:
		return thinkingBudgetMedium, true
	case providers.ReasoningEffortHigh:
		return thinkingBudgetHigh, true
	case providers.ReasoningEffortXHigh, providers.ReasoningEffortMax:
		return maximalThinkingBudget(model), true
	default:
		return 0, false
	}
}

func minimalThinkingBudget(model string) int32 {
	switch {
	case strings.Contains(model, "flash-lite"):
		return thinkingBudgetMinimalLite
	case strings.Contains(model, "pro"):
		return thinkingBudgetProMinimal
	default:
		return thinkingBudgetMinimal
	}
}

func maximalThinkingBudget(model string) int32 {
	if strings.Contains(model, "pro") {
		return thinkingBudgetMax
	}
	return thinkingBudgetHigh
}

func usesThinkingLevel(model string) bool {
	return strings.HasPrefix(geminiModelName(model), "gemini-3")
}

func usesThinkingBudget(model string) bool {
	model = geminiModelName(model)
	return strings.HasPrefix(model, "gemini-2.5") || strings.HasPrefix(model, "robotics-er-1.6")
}

func geminiModelName(model string) string {
	if slash := strings.LastIndexByte(model, '/'); slash >= 0 {
		model = model[slash+1:]
	}
	return strings.ToLower(model)
}
