package sdk

import (
	"os/exec"
	"regexp"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestVersionFormat validates the Version const matches the semver pattern.
func TestVersionFormat(t *testing.T) {
	t.Parallel()

	versionPattern := `^v\d+\.\d+\.\d+$`
	regex := regexp.MustCompile(versionPattern)
	require.True(t, regex.MatchString(Version))
}

// TestVersionMatchesLatestTag validates the Version const matches the latest git tag.
func TestVersionMatchesLatestTag(t *testing.T) {
	t.Parallel()

	cmd := exec.Command("git", "describe", "--tags", "--abbrev=0")
	output, err := cmd.Output()
	if err != nil {
		t.Skip("no git tags available")
	}

	expected := strings.TrimSpace(string(output))
	require.Equal(t, expected, Version)
}
