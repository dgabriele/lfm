"use client";

import { Disclosure, DisclosureGroup } from "@heroui/react";
import { ChevronDown } from "lucide-react";
import { z } from "zod";
import {
  getFieldMeta,
  type FieldMeta,
  type SectionDef,
} from "@/lib/config-schemas/meta";
import {
  Field,
  TextInput,
  NumberInput,
  SelectInput,
  BoolSwitch,
  IntListInput,
} from "./inputs";

/**
 * Schema-driven form.
 *
 * Takes a Zod object schema whose leaf fields carry `FieldMeta` via
 * `.meta()`.  Groups fields by `meta.section`, renders each section as
 * a HeroUI `Disclosure` (accordion item), and renders one input per
 * leaf based on `meta.inputKind`.
 *
 * To stay DRY across config types: adding a new field = one line in
 * the schema.  Adding a new input kind = one case in `renderInput`.
 * Dynamic option sources (e.g. "list of corpora") plug in via the
 * `optionProviders` prop so the component itself never imports data
 * access code.
 */

export type OptionProviders = Record<
  string,
  readonly { value: string; label: string }[]
>;

type SchemaObject = z.ZodObject<z.ZodRawShape>;

export function SchemaForm<T extends Record<string, unknown>>({
  schema,
  sections,
  value,
  onChange,
  optionProviders = {},
}: {
  schema: SchemaObject;
  sections: readonly SectionDef[];
  value: T;
  onChange: (next: T) => void;
  optionProviders?: OptionProviders;
}) {
  // Bucket fields by their declared section, preserving schema order.
  type Entry = { key: keyof T & string; field: z.ZodType; meta: FieldMeta };
  const bySection: Record<string, Entry[]> = {};
  for (const [key, field] of Object.entries(schema.shape)) {
    const meta = getFieldMeta(field);
    if (!meta || meta.hidden) continue;
    (bySection[meta.section] ??= []).push({
      key: key as keyof T & string,
      field,
      meta,
    });
  }

  const setField = <K extends keyof T>(key: K, next: T[K]) =>
    onChange({ ...value, [key]: next });

  return (
    <DisclosureGroup
      defaultExpandedKeys={[sections[0]?.key]}
      className="flex flex-col gap-2"
    >
      {sections
        .filter((s) => (bySection[s.key]?.length ?? 0) > 0)
        .map((section) => (
          <Disclosure
            key={section.key}
            id={section.key}
            className="rounded-[calc(var(--radius)*0.7)] border border-separator bg-surface/40"
          >
            <Disclosure.Heading>
              <Disclosure.Trigger className="w-full flex items-center justify-between gap-3 px-4 py-3 text-sm font-semibold text-accent/90 hover:text-accent transition-colors">
                <span className="uppercase tracking-wider text-xs">
                  {section.label}
                </span>
                <Disclosure.Indicator>
                  <ChevronDown
                    className="w-4 h-4 transition-transform data-[expanded]:rotate-180"
                    strokeWidth={2}
                  />
                </Disclosure.Indicator>
              </Disclosure.Trigger>
            </Disclosure.Heading>
            <Disclosure.Content>
              <Disclosure.Body className="grid grid-cols-1 lg:grid-cols-2 gap-5 px-4 pb-4 pt-1">
                {bySection[section.key]!.map(({ key, meta }) => (
                  <Field
                    key={key}
                    label={meta.label}
                    caption={meta.caption}
                  >
                    {renderInput(
                      key,
                      meta,
                      value[key],
                      (v) => setField(key, v as T[typeof key]),
                      optionProviders,
                    )}
                  </Field>
                ))}
              </Disclosure.Body>
            </Disclosure.Content>
          </Disclosure>
        ))}
    </DisclosureGroup>
  );
}

function renderInput(
  key: string,
  meta: FieldMeta,
  value: unknown,
  set: (v: unknown) => void,
  optionProviders: OptionProviders,
): React.ReactNode {
  switch (meta.inputKind) {
    case "number":
      return (
        <NumberInput
          value={typeof value === "number" ? value : 0}
          onValueChange={set}
          step={meta.step}
          min={meta.min}
          max={meta.max}
          name={key}
        />
      );
    case "bool":
      return (
        <BoolSwitch
          value={Boolean(value)}
          onValueChange={set}
        />
      );
    case "select":
      return (
        <SelectInput
          value={String(value ?? "")}
          onValueChange={set}
          options={meta.options ?? []}
          name={key}
        />
      );
    case "corpus-select":
      return (
        <SelectInput
          value={String(value ?? "")}
          onValueChange={set}
          options={optionProviders.corpora ?? []}
          name={key}
        />
      );
    case "int-list":
      return (
        <IntListInput
          value={Array.isArray(value) ? (value as number[]) : []}
          onValueChange={set}
        />
      );
    case "path":
    case "text":
    default:
      return (
        <TextInput
          value={typeof value === "string" ? value : ""}
          onChange={(e) => set(e.target.value)}
          name={key}
          spellCheck={false}
        />
      );
  }
}
