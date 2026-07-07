WITH catalog_dedup AS (
    SELECT
        regexp_replace(lower(document_name), '\s+', '', 'g') AS document_name_key,
        MIN(lark_created_at) AS lark_created_at,
        MAX(COALESCE(lark_updated_at, lark_created_at)) AS lark_updated_at
    FROM lark_document_catalog
    WHERE NOT is_deleted
      AND document_name <> ''
    GROUP BY regexp_replace(lower(document_name), '\s+', '', 'g')
)
UPDATE ingestion_registry AS registry
SET created_at = COALESCE(catalog_dedup.lark_created_at, registry.created_at),
    updated_at = COALESCE(catalog_dedup.lark_updated_at, registry.updated_at)
FROM catalog_dedup
WHERE regexp_replace(lower(registry.document_name), '\s+', '', 'g')
      = catalog_dedup.document_name_key
  AND (
      registry.created_at IS DISTINCT FROM COALESCE(
          catalog_dedup.lark_created_at,
          registry.created_at
      )
      OR registry.updated_at IS DISTINCT FROM COALESCE(
          catalog_dedup.lark_updated_at,
          registry.updated_at
      )
  );
