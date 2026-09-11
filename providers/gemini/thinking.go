package gemini

import (
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"google.golang.org/genai"

	"github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

const (
	// Gemini 2.5 exposes token ranges rather than named efforts. The Python
	// binding maps minimal and low to 1024 and caps known models at their maximum.
	thinkingBudgetHigh   int32 = 24576
	thinkingBudgetLow    int32 = 1024
	thinkingBudgetMax    int32 = 32768
	thinkingBudgetMedium int32 = 8192
)

var (
	geminiVersionPattern = regexp.MustCompile(`(?:^|/)gemini-(\d+)(?:\.(\d+))?`)
	numericSuffixPattern = regexp.MustCompile(`^(?:-\d+)+$`)

	allThinkingLevels = []genai.ThinkingLevel{
		genai.ThinkingLevelMinimal,
		genai.ThinkingLevelLow,
		genai.ThinkingLevelMedium,
		genai.ThinkingLevelHigh,
	}
	// Known Gemini 3 capabilities refine the permissive version routing used
	// for custom, dated, and newly released model IDs.
	// https://ai.google.dev/gemini-api/docs/generate-content/thinking#thinking-levels
	thinkingLevelsByModel = map[string][]genai.ThinkingLevel{
		"gemini-3.8-flash": {
			genai.ThinkingLevelLow, genai.ThinkingLevelMedium, genai.ThinkingLevelHigh,
		},
		"gemini-3.7-flash": {
			genai.ThinkingLevelLow, genai.ThinkingLevelMedium, genai.ThinkingLevelHigh,
		},
		"gemini-3.6-flash":      allThinkingLevels,
		"gemini-3.5-flash":      allThinkingLevels,
		"gemini-3.5-flash-lite": allThinkingLevels,
		"gemini-3.1-flash-lite": allThinkingLevels,
		"gemini-3.1-pro-preview": {
			genai.ThinkingLevelLow,
			genai.ThinkingLevelMedium,
			genai.ThinkingLevelHigh,
		},
		"gemini-3.1-flash-image":      {genai.ThinkingLevelMinimal, genai.ThinkingLevelHigh},
		"gemini-3.1-flash-lite-image": {genai.ThinkingLevelMinimal, genai.ThinkingLevelHigh},
		"gemini-3-flash-preview":      allThinkingLevels,
	}
	maxThinkingBudgetByModel = map[string]int32{
		"gemini-2.5-pro":        thinkingBudgetMax,
		"gemini-2.5-flash":      thinkingBudgetHigh,
		"gemini-2.5-flash-lite": thinkingBudgetHigh,
	}
)

// applyThinking preserves SDK defaults unless the caller explicitly selects an effort.
func applyThinking(cfg *genai.GenerateContentConfig, model string, effort providers.ReasoningEffort) error {
	if effort == "" || effort == providers.ReasoningEffortAuto {
		return nil
	}

	modelName := geminiModelName(model)
	if effort == providers.ReasoningEffortNone {
		if usesThinkingLevel(model) || matchesKnownModel(modelName, "gemini-2.5-pro") {
			return fmt.Errorf("model %q rejects effort %q: %w", model, effort,
				errors.NewUnsupportedParamError(providerName, "reasoning_effort"))
		}
		cfg.ThinkingConfig = &genai.ThinkingConfig{ThinkingBudget: new(int32)}
		return nil
	}

	if supportedLevels := knownThinkingLevels(modelName); supportedLevels != nil || usesThinkingLevel(model) {
		level, ok := thinkingLevel(effort)
		// Google's OpenAI compatibility contract maps minimal to low for Gemini 3.1 Pro.
		if matchesKnownModel(modelName, "gemini-3.1-pro-preview") && effort == providers.ReasoningEffortMinimal {
			level, ok = genai.ThinkingLevelLow, true
		}
		if !ok || supportedLevels != nil && !slices.Contains(supportedLevels, level) {
			return fmt.Errorf("model %q rejects effort %q: %w", model, effort,
				errors.NewUnsupportedParamError(providerName, "reasoning_effort"))
		}
		cfg.ThinkingConfig = &genai.ThinkingConfig{IncludeThoughts: true, ThinkingLevel: level}
		return nil
	}

	budget, ok := thinkingBudget(effort)
	if !ok {
		return fmt.Errorf("model %q rejects effort %q: %w", model, effort,
			errors.NewUnsupportedParamError(providerName, "reasoning_effort"))
	}
	if maximum := knownMaxThinkingBudget(modelName); maximum != 0 {
		budget = min(budget, maximum)
	}
	cfg.ThinkingConfig = &genai.ThinkingConfig{IncludeThoughts: true, ThinkingBudget: &budget}
	return nil
}

func geminiModelName(model string) string {
	if slash := strings.LastIndexByte(model, '/'); slash >= 0 {
		model = model[slash+1:]
	}
	return strings.ToLower(model)
}

func knownMaxThinkingBudget(modelName string) int32 {
	for knownModel, maximum := range maxThinkingBudgetByModel {
		if matchesKnownModel(modelName, knownModel) {
			return maximum
		}
	}
	return 0
}

func knownThinkingLevels(modelName string) []genai.ThinkingLevel {
	for knownModel, supported := range thinkingLevelsByModel {
		if matchesKnownModel(modelName, knownModel) {
			return supported
		}
	}
	return nil
}

func matchesKnownModel(modelName, knownModel string) bool {
	if modelName == knownModel {
		return true
	}
	return strings.HasPrefix(modelName, knownModel) &&
		numericSuffixPattern.MatchString(strings.TrimPrefix(modelName, knownModel))
}

func thinkingBudget(effort providers.ReasoningEffort) (int32, bool) {
	switch effort {
	case providers.ReasoningEffortMinimal, providers.ReasoningEffortLow:
		return thinkingBudgetLow, true
	case providers.ReasoningEffortMedium:
		return thinkingBudgetMedium, true
	case providers.ReasoningEffortHigh:
		return thinkingBudgetHigh, true
	case providers.ReasoningEffortXHigh, providers.ReasoningEffortMax:
		return thinkingBudgetMax, true
	default:
		return 0, false
	}
}

func thinkingLevel(effort providers.ReasoningEffort) (genai.ThinkingLevel, bool) {
	switch effort {
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

func usesThinkingLevel(model string) bool {
	match := geminiVersionPattern.FindStringSubmatch(strings.ToLower(model))
	if match == nil {
		return false
	}
	major, err := strconv.Atoi(match[1])
	return err == nil && major >= 3
}
