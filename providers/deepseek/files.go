package deepseek

import (
	"context"
	stderrors "errors"

	llmerrors "github.com/mozilla-ai/any-llm-go/errors"
	"github.com/mozilla-ai/any-llm-go/providers"
)

const (
	minFileExpiration = 3600
	maxFileExpiration = 2592000
)

// UploadFile uploads an image for reuse by DeepSeek Vision requests.
func (p *Provider) UploadFile(ctx context.Context, params providers.UploadFileParams) (*providers.File, error) {
	if params.File == nil {
		return nil, invalidFileRequest("file is required")
	}
	if params.Purpose != providers.FilePurposeUserData {
		return nil, invalidFileRequest("purpose must be user_data")
	}
	if params.ExpiresAfter != nil &&
		(*params.ExpiresAfter < minFileExpiration || *params.ExpiresAfter > maxFileExpiration) {
		return nil, invalidFileRequest("expires_after must be between 3600 and 2592000 seconds")
	}
	return p.CompatibleProvider.UploadFile(ctx, params)
}

// ListFiles lists one page of files uploaded for DeepSeek Vision requests.
func (p *Provider) ListFiles(ctx context.Context, opts providers.ListFilesOptions) (*providers.FileList, error) {
	if opts.Limit != nil && (*opts.Limit < 1 || *opts.Limit > 1000) {
		return nil, invalidFileRequest("limit must be between 1 and 1000")
	}
	if opts.Order != "" && opts.Order != providers.FileOrderAsc && opts.Order != providers.FileOrderDesc {
		return nil, invalidFileRequest("order must be asc or desc")
	}
	if opts.Purpose != "" && opts.Purpose != providers.FilePurposeUserData {
		return nil, invalidFileRequest("purpose must be user_data")
	}
	return p.CompatibleProvider.ListFiles(ctx, opts)
}

// RetrieveFile returns metadata for one DeepSeek file.
func (p *Provider) RetrieveFile(ctx context.Context, fileID string) (*providers.File, error) {
	if fileID == "" {
		return nil, invalidFileRequest("file_id is required")
	}
	return p.CompatibleProvider.RetrieveFile(ctx, fileID)
}

// DeleteFile deletes one DeepSeek file.
func (p *Provider) DeleteFile(ctx context.Context, fileID string) (*providers.DeletedFile, error) {
	if fileID == "" {
		return nil, invalidFileRequest("file_id is required")
	}
	return p.CompatibleProvider.DeleteFile(ctx, fileID)
}

func invalidFileRequest(message string) error {
	return llmerrors.NewInvalidRequestError(providerName, stderrors.New(message))
}
