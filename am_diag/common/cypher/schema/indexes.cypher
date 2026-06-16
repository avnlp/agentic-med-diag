// Performance indexes for hot query paths.
CREATE INDEX entity_label IF NOT EXISTS FOR (n:Entity) ON (n.label);
CREATE RANGE INDEX rel_id IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.id);
CREATE RANGE INDEX rel_type IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.type);
