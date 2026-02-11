package sdk

import (
	"regexp"
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
